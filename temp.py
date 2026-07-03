import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
from boxmot.trackers.tracker_zoo import create_tracker
from rfdetr import RFDETR2XLarge
from PIL import Image
import torch
from boxmot.reid import ReID
from scipy.optimize import linear_sum_assignment

# ==========================================
# ⚙️ КОНФИГУРАЦИЯ
# ==========================================
VIDEO_REF = "/mnt/c/Coding/hf_20260526_192100_c64bed19-1a67-440a-b443-66c72e51aa6d.mp4"
VIDEO_TARGET = "/mnt/c/Coding/hf_20260526_175347_c14339e3-ab87-4c90-987a-f24e2aae63bd.mp4"
TABLE_JSON = "tables_map.json"
TARGET_RES = (1920, 1088)
MODEL_SIZE = 1920
TABLE_THRESH = 100
FPS_OVERRIDE = 30
OUTPUT_FILE = "output_processed.mp4"
LOG_FILE = "output_events.txt"

def print_progress(current, total, stage_name=""):
    percent = (current / total) * 100
    sys.stdout.write(f'\r  {stage_name}: [{current}/{total}] {percent:.1f}% ')
    sys.stdout.flush()
    if current == total:
        print()

def assign_table_ids(boxes):
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    result = []
    for i, box in enumerate(sorted_boxes, start=1):
        result.append({'id': i, 'box': box})
    return result

class EventLogger:
    def __init__(self, filepath, video_path, fps, target_res):
        self.filepath = filepath
        self.video_path = os.path.basename(video_path)
        self.fps = fps
        self.target_res = target_res
        self.events = []
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📋 ОТЧЕТ ОБРАБОТКИ ВИДЕО: DETECTION + TRACKING + EVENTS\n")
            f.write("="*80 + "\n")
            f.write(f"Видео: {self.video_path}\n")
            f.write(f"Разрешение: {target_res[0]}x{target_res[1]}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Порог дистанции до стола: {TABLE_THRESH}px\n")
            f.write(f"Мин. длительность события: 3 сек ({int(fps*3)} кадров)\n")
            f.write(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-"*80 + "\n\n")
            f.write("📌 СОБЫТИЯ:\n")
            f.write("-"*80 + "\n")

    def log_event(self, frame, timestamp_sec, tid, role, posture, near_table, 
                  bbox, table_id, table_dist, msg):
        entry = {
            'frame': frame, 'time_sec': timestamp_sec,
            'time_str': f"{int(timestamp_sec//3600):02d}:{int((timestamp_sec%3600)//60):02d}:{int(timestamp_sec%60):02d}",
            'track_id': tid, 'role': role, 'posture': posture, 'near_table': near_table,
            'table_id': table_id, 'bbox': [int(b) for b in bbox],
            'table_dist_px': round(table_dist, 1), 'message': msg
        }
        self.events.append(entry)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"[{entry['time_str']}] Frame {frame:05d} | ID:{tid:03d} | {role:8s} | {posture:8s}\n")
            f.write(f"  📍 Координаты: {entry['bbox']}\n")
            near_str = "✅ БЛИЗКО" if near_table else "❌ ДАЛЕКО"
            f.write(f"  🪑 Стол #{entry['table_id']} | Дистанция: {entry['table_dist_px']}px {near_str}\n")
            f.write(f"  💬 {msg}\n")
            f.write("-"*80 + "\n")

    def finalize(self, total_frames, processing_time_sec):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n📊 ИТОГОВАЯ СТАТИСТИКА:\n" + "="*80 + "\n")
            f.write(f"Всего кадров: {total_frames} | Длительность: {total_frames/self.fps:.1f} сек\n")
            f.write(f"Время обработки: {processing_time_sec:.1f} сек | Скорость: {total_frames/processing_time_sec:.2f} FPS\n")
            f.write(f"Всего событий: {len(self.events)}\n")
            if self.events:
                f.write("\n📈 Распределение событий:\n")
                stats = {}
                for e in self.events:
                    key = e['message'].split(']')[-1].strip().split('(')[0].strip()
                    stats[key] = stats.get(key, 0) + 1
                for k, v in sorted(stats.items(), key=lambda x: -x[1]): f.write(f"  • {k}: {v}\n")
                f.write("\n🪑 Активность по столам:\n")
                table_stats = {}
                for e in self.events:
                    tid = e['table_id']
                    table_stats[tid] = table_stats.get(tid, 0) + 1
                for tid, cnt in sorted(table_stats.items(), key=lambda x: -x[1]): f.write(f"  • Стол #{tid}: {cnt} событий\n")
            f.write("\n✅ Обработка завершена.\n")
        print(f"📝 Лог событий сохранён: {self.filepath}")

def ensemble_detections(yolo_res, rfdetr_res, target_classes, conf_thresh=0.3, iou_thresh=0.5):
    boxes_list, confs_list, classes_list = [], [], []
    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
        data = yolo_res.boxes.data.cpu().numpy()
        mask = (data[:, 4] >= conf_thresh) & np.isin(data[:, 5], target_classes)
        if np.any(mask):
            boxes_list.append(data[mask, :4]); confs_list.append(data[mask, 4]); classes_list.append(data[mask, 5])
    if len(rfdetr_res) > 0:
        mask = (rfdetr_res.confidence >= conf_thresh) & np.isin(rfdetr_res.class_id, target_classes)
        if np.any(mask):
            boxes_list.append(rfdetr_res.xyxy[mask]); confs_list.append(rfdetr_res.confidence[mask]); classes_list.append(rfdetr_res.class_id[mask])
    if not boxes_list: return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
    all_boxes = np.vstack(boxes_list); all_confs = np.concatenate(confs_list); all_classes = np.concatenate(classes_list)
    final_boxes, final_confs, final_classes = [], [], []
    for cls in np.unique(all_classes):
        cls_mask = all_classes == cls; b, s = all_boxes[cls_mask].tolist(), all_confs[cls_mask].tolist()
        indices = cv2.dnn.NMSBoxes(b, s, conf_thresh, iou_thresh)
        if indices is None or len(indices) == 0: continue
        idx = np.array(indices).flatten()
        final_boxes.append(all_boxes[cls_mask][idx]); final_confs.append(all_confs[cls_mask][idx]); final_classes.append(np.full(len(idx), cls, dtype=np.float32))
    if not final_boxes: return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
    return np.vstack(final_boxes), np.concatenate(final_confs), np.concatenate(final_classes)

def classify_role(keypoints, bbox):
    x1, y1, x2, y2 = bbox; h, w = y2-y1, x2-x1
    if keypoints is None or len(keypoints) == 0: return "Customer" if h/w < 1.6 else "Waiter"
    ls, rs = keypoints[5], keypoints[6]; lh, rh = keypoints[11], keypoints[12]; lk, rk = keypoints[13], keypoints[14]
    if lh[2]>0.4 and lk[2]>0.4 and rh[2]>0.4 and rk[2]>0.4:
        def angle(p1, p2, p3):
            v1, v2 = np.array(p1[:2])-np.array(p2[:2]), np.array(p3[:2])-np.array(p2[:2])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2+1e-6), -1, 1)))
        return "Customer" if (angle(ls, lh, lk) + angle(rs, rh, rk))/2 < 135 else "Waiter"
    return "Customer" if h/w < 1.6 else "Waiter"

def get_posture(keypoints):
    if keypoints is None: return "Standing"
    lh, rh, lk, rk = keypoints[11], keypoints[12], keypoints[13], keypoints[14]
    if lh[2]>0.4 and lk[2]>0.4 and rh[2]>0.4 and rk[2]>0.4:
        def angle(p1, p2, p3):
            v1, v2 = np.array(p1[:2])-np.array(p2[:2]), np.array(p3[:2])-np.array(p2[:2])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2+1e-6), -1, 1)))
        return "Sitting" if (angle(keypoints[5], lh, lk) + angle(keypoints[6], rh, rk))/2 < 135 else "Standing"
    return "Standing"

def dist_to_nearest_table_with_id(bbox, table_list):
    if not table_list: return float('inf'), None
    cx = (bbox[0] + bbox[2]) / 2.0; cy = (bbox[1] + bbox[3]) / 2.0
    min_dist, nearest_id = float('inf'), None
    for t in table_list:
        x1, y1, x2, y2 = t['box']
        dx = np.maximum(x1 - cx, 0.0) + np.maximum(cx - x2, 0.0)
        dy = np.maximum(y1 - cy, 0.0) + np.maximum(cy - y2, 0.0)
        dist = np.hypot(dx, dy)
        if dist < min_dist: min_dist, nearest_id = dist, t['id']
    return min_dist, nearest_id

class TrackerEnsemble:
    def __init__(self, trackers, iou_thresh=0.5):
        self.trackers, self.iou_thresh = trackers, iou_thresh
        self.global_id_map, self.next_global_id, self.last_seen, self.prev_boxes = {}, 1, {}, {}
    def update(self, dets, frame, frame_idx):
        all_tracks = [tr.update(dets, frame) for tr in self.trackers]
        combined = []
        for idx, tracks in enumerate(all_tracks):
            if len(tracks)==0: continue
            for t in tracks: combined.append({'box': t[:4], 'id': int(t[4]), 'conf': t[5] if len(t)>5 else 0.8, 'tr_idx': idx})
        if not combined: self._cleanup(frame_idx); return np.empty((0, 6))
        boxes = np.array([c['box'] for c in combined])
        iou_cost = 1.0 - sv.box_iou_batch(boxes, boxes); np.fill_diagonal(iou_cost, 1.0)
        row_ind, col_ind = linear_sum_assignment(iou_cost)
        used, clusters = set(), []
        for r, c in zip(row_ind, col_ind):
            if iou_cost[r,c] < 0.55 and r not in used and c not in used: clusters.append([r, c]); used.update([r, c])
        for i in range(len(boxes)):
            if i not in used: clusters.append([i])
        final_tracks = []
        for cluster in clusters:
            items = [combined[i] for i in cluster]
            fused = np.mean([i['box'] for i in items], axis=0)
            primary = next((i for i in items if i['tr_idx']==0), items[0])
            k = (primary['tr_idx'], primary['id'])
            if k not in self.global_id_map: self.global_id_map[k] = self.next_global_id; self.next_global_id += 1
            gid = self.global_id_map[k]; self.last_seen[gid] = frame_idx; self.prev_boxes[gid] = fused.copy()
            final_tracks.append([*fused, gid, np.mean([i['conf'] for i in items])])
        self._cleanup(frame_idx); return np.array(final_tracks) if final_tracks else np.empty((0, 6))
    def _cleanup(self, frame_idx, max_age=120):
        for g in [k for k, v in self.last_seen.items() if frame_idx - v > max_age]:
            del self.last_seen[g]; self.global_id_map.pop((0,g), None); self.global_id_map.pop((1,g), None); self.prev_boxes.pop(g, None)

# ==========================================
# 🟢 ФАЗА 1: Извлечение столов
# ==========================================
print("="*50); print("🔍 ФАЗА 1: Сканирование референсного видео..."); print("="*50)
cap_ref = cv2.VideoCapture(VIDEO_REF)
if not cap_ref.isOpened(): raise RuntimeError(f"❌ Не удалось открыть {VIDEO_REF}")
ref_w, ref_h = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
sx_ref, sy_ref = TARGET_RES[0]/ref_w, TARGET_RES[1]/ref_h
table_model = YOLO("yolo26x.pt")
all_ref_boxes, frame_cnt = [], 0
total_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = cap_ref.read()
    if not ret: break
    frame_cnt += 1
    if frame_cnt % 30 != 0: continue
    frame_res = cv2.resize(frame, TARGET_RES)
    res = table_model(frame_res, conf=0.25, classes=[60], verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        d = res.boxes.data.cpu().numpy()
        mask = d[:, 4] > 0.4
        if mask.any():
            orig = d[mask, :4].copy()
            orig[:, [0,2]] /= sx_ref; orig[:, [1,3]] /= sy_ref
            all_ref_boxes.extend(orig.tolist())
    if frame_cnt % 300 == 0: print_progress(frame_cnt, total_ref, "🔍 Поиск столов")
cap_ref.release()
print_progress(frame_cnt, total_ref, "🔍 Поиск столов")

unique_tables = []
if all_ref_boxes:
    idx = cv2.dnn.NMSBoxes(all_ref_boxes, [0.8]*len(all_ref_boxes), 0.3, 0.5)
    unique_boxes = [all_ref_boxes[i] for i in idx.flatten()] if len(idx)>0 else []
    unique_tables = assign_table_ids(unique_boxes)

with open(TABLE_JSON, 'w') as f: 
    json.dump({"source_resolution": [ref_w, ref_h], "tables": unique_tables}, f, indent=2)
print(f"✅ Найдено {len(unique_tables)} уникальных столов в {TABLE_JSON}")

# ==========================================
# 🟢 ФАЗА 2: Инициализация
# ==========================================
print("\n" + "="*50); print(f"📋 ФАЗА 2: Инициализация моделей"); print("="*50)
with open(TABLE_JSON) as f: map_data = json.load(f)
ref_res = map_data.get("source_resolution", TARGET_RES)
sx, sy = TARGET_RES[0]/ref_res[0], TARGET_RES[1]/ref_res[1]
static_tables = [{'id': t['id'], 'box': [int(t['box'][0]*sx), int(t['box'][1]*sy), int(t['box'][2]*sx), int(t['box'][3]*sy)]} for t in map_data["tables"]]
print(f"📐 Масштабировано {len(static_tables)} столов под {TARGET_RES[0]}x{TARGET_RES[1]}")

det_model_people = RFDETR2XLarge(resolution=MODEL_SIZE, num_classes=90); det_model_people.optimize_for_inference()
det_model_tables = YOLO("yolo26x.pt"); pose_model = YOLO("yolo26x-pose.pt")

def make_reid_patcher(reid_obj, track_ref_dict):
    def get_features_patched(boxes, img):
        h, w = img.shape[:2]; n = len(boxes); out = np.zeros((n, 512), dtype=np.float32)
        if n==0: return out
        valid_idx, crops = [], []
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = map(int, b[:4]); crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size==0: continue
            q = min(1.0, (y2-y1)/70.0)*0.6 + max(0, 1.0-abs(((x2-x1)/(y2-y1))-0.45)/0.5)*0.3
            if q>0.45 and (y2-y1)>40: valid_idx.append(i); crops.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not valid_idx: return out
        infer_idx = [idx for idx in valid_idx if not any(np.hypot((boxes[idx][0]+boxes[idx][2])/2-((p[0]+p[2])/2), (boxes[idx][1]+boxes[idx][3])/2-((p[1]+p[3])/2))<5 for p in track_ref_dict.values())]
        if infer_idx:
            cmap = {i: crops[valid_idx.index(i)] for i in infer_idx}
            with torch.no_grad():
                r = reid_obj([cmap[i] for i in infer_idx])
                if torch.is_tensor(r): r = r.cpu().numpy()
            for k, i in enumerate(infer_idx):
                if k < len(r): out[i] = r[k]
        return out
    return get_features_patched

reid_boost = ReID(weights='osnet_ain_x1_0_msmt17.pt', device='cuda:0', half=True)
tr_boost = create_tracker(tracker_type='boosttrack', reid_model=None, device='cuda:0', half=False)
tr_boost.model = reid_boost; track_ref = {}; tr_boost.model.get_features = lambda b, i: make_reid_patcher(reid_boost, track_ref)(b, i)
reid_deep = ReID(weights='osnet_ain_x1_0_msmt17.pt', device='cuda:0', half=True)
tr_deep = create_tracker(tracker_type='deepocsort', reid_model=None, device='cuda:0', half=False)
tr_deep.model = reid_deep; tr_deep.model.get_features = lambda b, i: make_reid_patcher(reid_deep, track_ref)(b, i)

custom_params = {'track_thresh': 0.5, 'match_thresh': 0.5, 'track_buffer': 300, 'det_thresh': 0.5, 'iou_threshold': 0.5, 'new_track_thresh': 0.45, 'lambda_iou': 0.3, 'lambda_shape': 0.5, 'lambda_mhd': 0.2}
alt_map = {'track_thresh': ['det_thresh', 'track_high_thresh', 'conf_thresh'], 'match_thresh': ['iou_threshold', 'asso_thresh', 'match_threshold', 'reid_thresh', 'iou_thresh'], 'track_buffer': ['max_age', 'track_buffer_len', 'lost_track_buffer'], 'new_track_thresh': ['new_track_thresh', 'init_conf'], 'lambda_iou': ['lambda_iou', 'iou_weight'], 'lambda_shape': ['lambda_shape', 'shape_weight'], 'lambda_mhd': ['lambda_mhd', 'mahalanobis_weight']}
for tr in [tr_boost, tr_deep]:
    for pn, pv in custom_params.items():
        if hasattr(tr, pn): setattr(tr, pn, pv)
        else:
            for alt in alt_map.get(pn, []):
                if hasattr(tr, alt): setattr(tr, alt, pv); break

ensemble = TrackerEnsemble([tr_boost, tr_deep], iou_thresh=0.5)
print("✅ Все модели инициализированы.")

# ==========================================
# 🟢 ФАЗА 3: Обработка видео
# ==========================================
print("\n" + "="*50); print(f"🎬 ФАЗА 3: Обработка видео + логирование"); print("="*50)
pose_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)
person_states = {}
cap = cv2.VideoCapture(VIDEO_TARGET)
assert cap.isOpened(), f"❌ Не удалось открыть: {VIDEO_TARGET}"
fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS_OVERRIDE
total_target = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
MIN_DURATION_FRAMES = int(fps * 3.0)
logger = EventLogger(LOG_FILE, VIDEO_TARGET, fps, TARGET_RES)

print("🎞️ Инициализация VideoWriter...")
fourcc_codecs = ['avc1', 'H264', 'mp4v', 'XVID', 'MJPG']
out = None
for codec in fourcc_codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, TARGET_RES)
    if out.isOpened():
        print(f"✅ Кодек {codec} успешно инициализирован")
        break
if out is None or not out.isOpened():
    print("❌ Не удалось создать VideoWriter"); sys.exit(1)

frame_count, start_time = 0, time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 30 == 0: print_progress(frame_count, total_target, "🎞️ Обработка")
        if frame.shape[1] != TARGET_RES[0] or frame.shape[0] != TARGET_RES[1]:
            frame = cv2.resize(frame, TARGET_RES, interpolation=cv2.INTER_LANCZOS4)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        yolo_res = det_model_tables(frame, imgsz=MODEL_SIZE, conf=0.25, verbose=False)[0]
        rfdetr_res = det_model_people.predict(pil_img, threshold=0.25)
        all_boxes, all_confs, all_classes = ensemble_detections(yolo_res, rfdetr_res, target_classes=[0], conf_thresh=0.3, iou_thresh=0.5)
        person_boxes = all_boxes[all_classes == 0]; person_confs = all_confs[all_classes == 0]

        pose_results = pose_model(frame, imgsz=MODEL_SIZE, conf=0.5, verbose=False)[0]
        person_boxes_full, kpts_data = np.empty((0, 4)), np.empty((0, 17, 3))
        if pose_results.boxes is not None:
            pd = pose_results.boxes.data.cpu().numpy(); mask = pd[:, 5] == 0; person_boxes_full = pd[mask, :4]
            if pose_results.keypoints is not None:
                ka = pose_results.keypoints.data.cpu().numpy(); kpts_data = ka[mask] if len(ka)==len(pd) else np.empty((0, 17, 3))

        dets = np.column_stack((person_boxes, person_confs, np.zeros(len(person_boxes)))) if len(person_boxes)>0 else np.empty((0, 6))
        tracks = ensemble.update(dets, frame, frame_count)
        tracked_bboxes = tracks[:, :4] if len(tracks)>0 else np.empty((0, 4))
        track_ids = tracks[:, 4].astype(int) if len(tracks)>0 else np.empty((0,), dtype=int)
        track_ref = {tid: box for tid, box in zip(track_ids, tracked_bboxes)}

        assigned = []
        for bb in tracked_bboxes:
            if len(person_boxes_full)==0: assigned.append(None); continue
            ious = sv.box_iou_batch(np.expand_dims(bb, 0), person_boxes_full)[0]; idx = np.argmax(ious)
            assigned.append(kpts_data[idx] if ious[idx]>0.5 else None)

        roles = [classify_role(assigned[i], b) for i, b in enumerate(tracked_bboxes)]
        cur_ids = set()
        for i, (bb, tid) in enumerate(zip(tracked_bboxes, track_ids)):
            cur_ids.add(tid)
            role, kps = roles[i], assigned[i]; posture = get_posture(kps)
            table_dist, nearest_table_id = dist_to_nearest_table_with_id(bb, static_tables)
            near = table_dist < TABLE_THRESH
            state = (role, posture, near)
            if tid not in person_states: person_states[tid] = {'state': state, 'start': frame_count, 'met': False}
            st = person_states[tid]
            if st['state'] != state: st['state']=state; st['start']=frame_count; st['met']=False; continue
            if not st['met'] and (frame_count - st['start']) >= MIN_DURATION_FRAMES:
                st['met'] = True
                r,p,n = st['state']; msg=""
                if r=="Customer" and p=="Sitting" and n: msg=f"🪑 Customer (ID {tid}) sat at table #{nearest_table_id} (>3s)"
                elif r=="Customer" and p=="Standing" and n: msg=f"🚶 Customer (ID {tid}) stood up from table #{nearest_table_id} (>3s)"
                elif r=="Waiter" and n: msg=f"🍽 Waiter (ID {tid}) at table #{nearest_table_id} (>3s)"
                elif r=="Waiter" and not n: msg=f"🚶 Waiter (ID {tid}) left table #{nearest_table_id} (>3s)"
                if msg:
                    timestamp_sec = frame_count / fps
                    logger.log_event(frame=frame_count, timestamp_sec=timestamp_sec, tid=tid, role=role, posture=posture, near_table=near, bbox=bb, table_id=nearest_table_id or 0, table_dist=table_dist, msg=msg)
                    print(f"\n  📢 {msg}")
        for sid in list(person_states.keys()):
            if sid not in cur_ids: del person_states[sid]

        # 🔹 ВИЗУАЛИЗАЦИЯ (без оверлея событий в углу)
        ann = frame.copy()
        
        # 🪑 Столы с номерами
        for t in static_tables:
            x1,y1,x2,y2 = t['box']
            cv2.rectangle(ann, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(ann, f"#{t['id']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # 👥 Люди: рамки + подписи + скелет
        for i, (bb, tid) in enumerate(zip(tracked_bboxes, track_ids)):
            x1,y1,x2,y2 = bb.astype(int)
            cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)
            lab = f"ID: {tid} | {roles[i]}"
            (tw,th),_ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(ann, (x1, y1-30), (x1+tw+10, y1), (0,0,0), -1)
            cv2.putText(ann, lab, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            kps = assigned[i]
            if kps is not None:
                pts, cfs = kps[:,:2], kps[:,2]; msk = cfs>0.4
                if np.any(msk): ann = pose_annotator.annotate(ann, sv.KeyPoints(xy=pts[np.newaxis], confidence=cfs[np.newaxis]))

        # Запись кадра
        if ann.dtype != np.uint8: ann = np.clip(ann, 0, 255).astype(np.uint8)
        if len(ann.shape) == 3 and ann.shape[2] == 4: ann = cv2.cvtColor(ann, cv2.COLOR_BGRA2BGR)
        if ann.shape[:2] != (TARGET_RES[1], TARGET_RES[0]): ann = cv2.resize(ann, TARGET_RES, interpolation=cv2.INTER_LANCZOS4)
        out.write(ann)

finally:
    print_progress(frame_count, total_target, "🎞️ Обработка")
    cap.release(); out.release()
    elapsed = time.time() - start_time
    logger.finalize(total_frames=frame_count, processing_time_sec=elapsed)
    print(f"✅ Завершено за {elapsed:.1f} сек.")

if os.path.exists(OUTPUT_FILE):
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024*1024)
    print(f"🎬 Видео: {OUTPUT_FILE} ({TARGET_RES[0]}x{TARGET_RES[1]}, {size_mb:.1f} MB)")
if os.path.exists(LOG_FILE):
    log_size = os.path.getsize(LOG_FILE) / 1024
    print(f"📝 Лог: {LOG_FILE} ({log_size:.1f} KB, {len(logger.events)} событий)")
import argparse
import os
import time
import inspect
import importlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

try:
    import supervision as sv
    HAS_SUPERVISION = True
    SV_IMPORT_ERROR = None
except Exception as e:
    sv = None
    HAS_SUPERVISION = False
    SV_IMPORT_ERROR = e

try:
    import boxmot
    HAS_BOXMOT = True
    BOXMOT_IMPORT_ERROR = None
except Exception as e:
    boxmot = None
    HAS_BOXMOT = False
    BOXMOT_IMPORT_ERROR = e


# Классы, которые нам нужны.
CLASS_NAMES = ["person", "table"]
CLASS_COLORS = {
    0: (0, 220, 0),    # person: зелёный
    1: (255, 160, 0),  # table: оранжевый
}

TEXT_PROMPT = "person . table . desk . dining table ."


# Точные пути к трекерам для новой структуры boxmot 25.x
BOXMOT_TRACKER_PATHS = {
    "botsort": ("boxmot.trackers.box.botsort.tracker", "BotSort"),
    "bytetrack": ("boxmot.trackers.box.bytetrack.tracker", "ByteTrack"),
    "ocsort": ("boxmot.trackers.box.ocsort.tracker", "OcSort"),
    "deepocsort": ("boxmot.trackers.box.deepocsort.tracker", "DeepOcSort"),
    "strongsort": ("boxmot.trackers.box.strongsort.tracker", "StrongSort"),
    "hybridsort": ("boxmot.trackers.box.hybridsort.tracker", "HybridSort"),
    "occluboost": ("boxmot.trackers.box.occluboost.tracker", "OccluBoost"),
    "boosttrack": ("boxmot.trackers.box.boosttrack.tracker", "BoostTrack"),
    "sfsort": ("boxmot.trackers.box.sfsort.tracker", "SFSORT"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Детекция и трекинг людей и столов с использованием Grounding DINO."
    )

    parser.add_argument("--source", default="0", help="Путь к видео, RTSP или индекс камеры.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-size", type=int, default=1280)
    parser.add_argument("--output", default="")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--tracker",
        default="auto",
        choices=[
            "auto",
            "botsort",
            "bytetrack",
            "ocsort",
            "deepocsort",
            "strongsort",
            "hybridsort",
            "occluboost",
            "boosttrack",
            "sfsort",
            "simple",
        ],
        help="Выбор трекера. auto: botsort -> bytetrack -> ocsort -> simple.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Сколько кадров хранить пропавший трек (track_buffer).",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument(
        "--reid-weights",
        default="osnet_x1_0_msmt17.pt",
        help="Путь к весам ReID-модели для трекеров, которые используют ReID.",
    )

    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()
    return model, processor


def resize_for_inference(frame: np.ndarray, max_size: int):
    h, w = frame.shape[:2]

    if max_size is None or max_size <= 0:
        return frame, 1.0

    longest = max(h, w)
    if longest <= max_size:
        return frame, 1.0

    scale = max_size / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, scale


def decode_labels(labels, processor):
    if labels is None:
        return []

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

        if labels.dtype in (torch.long, torch.int):
            if labels.dim() == 0:
                return [processor.tokenizer.decode(int(labels), skip_special_tokens=True)]

            if labels.dim() == 1:
                return [
                    processor.tokenizer.decode(int(x), skip_special_tokens=True)
                    for x in labels
                ]

            return [
                processor.tokenizer.decode(row.tolist(), skip_special_tokens=True)
                for row in labels
            ]

        return labels.tolist()

    if isinstance(labels, np.ndarray):
        return labels.tolist()

    if isinstance(labels, (list, tuple)):
        if len(labels) > 0 and isinstance(labels[0], (list, tuple)):
            return list(labels[0])
        return list(labels)

    return [labels]


def normalize_label(text: str) -> str:
    t = (
        str(text)
        .lower()
        .strip()
        .replace(".", " ")
        .replace(",", " ")
        .replace("_", " ")
        .strip()
    )

    person_keywords = (
        "person",
        "people",
        "human",
        "pedestrian",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "worker",
    )

    table_keywords = (
        "table",
        "desk",
        "dining table",
        "office table",
        "counter",
    )

    if any(k in t for k in person_keywords):
        return "person"

    if any(k in t for k in table_keywords):
        return "table"

    return t


def apply_nms(boxes, scores, class_ids, iou_threshold):
    if len(boxes) == 0 or iou_threshold <= 0:
        return boxes, scores, class_ids

    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    cls_t = torch.from_numpy(class_ids.astype(np.int64))

    offset = cls_t * 1_000_000.0
    boxes_for_nms = boxes_t + offset.unsqueeze(1)

    keep = nms(boxes_for_nms, scores_t, iou_threshold).cpu().numpy()

    return boxes[keep], scores[keep], class_ids[keep]


def detect_objects(model, processor, frame_bgr, device, args):
    infer_img, scale = resize_for_inference(frame_bgr, args.max_size)
    h, w = infer_img.shape[:2]

    rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    inputs = processor(images=pil_image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    use_amp = args.fp16 and device.type == "cuda"

    if use_amp:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
    else:
        with torch.inference_mode():
            outputs = model(**inputs)

    postprocess_fn = processor.post_process_grounded_object_detection
    pp_params = inspect.signature(postprocess_fn).parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in pp_params.values()
    )

    pp_kwargs = {}
    used_target_sizes = False

    if has_var_kwargs or "target_sizes" in pp_params:
        pp_kwargs["target_sizes"] = [(h, w)]
        used_target_sizes = True

    if "box_threshold" in pp_params:
        pp_kwargs["box_threshold"] = args.box_threshold
    elif "threshold" in pp_params:
        pp_kwargs["threshold"] = args.box_threshold
    elif "score_threshold" in pp_params:
        pp_kwargs["score_threshold"] = args.box_threshold
    elif has_var_kwargs:
        pp_kwargs["threshold"] = args.box_threshold

    if "text_threshold" in pp_params or has_var_kwargs:
        pp_kwargs["text_threshold"] = args.text_threshold

    results = postprocess_fn(outputs, inputs["input_ids"], **pp_kwargs)[0]

    boxes = results["boxes"].detach().float().cpu().numpy()
    scores = results["scores"].detach().float().cpu().numpy()

    if "text_labels" in results and results["text_labels"] is not None:
        labels_raw = results["text_labels"]
    else:
        labels_raw = results.get("labels", [])

    if not used_target_sizes:
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)

    labels = decode_labels(labels_raw, processor)
    if len(labels) == 1 and len(boxes) > 1:
        labels = labels * len(boxes)

    n = min(len(boxes), len(scores), len(labels))
    boxes, scores, labels = boxes[:n], scores[:n], labels[:n]

    normalized_labels = [normalize_label(x) for x in labels]
    keep = [i for i, label in enumerate(normalized_labels) if label in CLASS_NAMES]

    if len(keep) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = boxes[keep] / scale
    scores = scores[keep]
    labels = [normalized_labels[i] for i in keep]
    class_ids = np.array([CLASS_NAMES.index(label) for label in labels], dtype=np.int32)

    H, W = frame_bgr.shape[:2]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    return (
        boxes[valid].astype(np.float32),
        scores[valid].astype(np.float32),
        class_ids[valid].astype(np.int32),
    )


class SimpleIoUTracker:
    def __init__(self, max_age=30, iou_threshold=0.25, min_new_score=0.2):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_new_score = min_new_score

    @staticmethod
    def _iou_matrix(boxes_a, boxes_b):
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

        ax1, ay1, ax2, ay2 = boxes_a.T
        bx1, by1, bx2, by2 = boxes_b.T

        inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
        inter_y1 = np.maximum(ay1[:, None], by1[None, :])
        inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
        inter_y2 = np.minimum(ay2[:, None], by2[None, :])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        union_area = area_a[:, None] + area_b[None, :] - inter_area

        return np.where(union_area > 0, inter_area / union_area, 0.0).astype(np.float32)

    def update(self, boxes, class_ids, scores, frame=None):
        for track in self.tracks:
            track["missed"] += 1

        matched_det = set()

        if len(boxes) > 0:
            for cls_id in np.unique(class_ids):
                det_idx = np.where(class_ids == cls_id)[0]
                track_idx = [
                    i for i, track in enumerate(self.tracks)
                    if track["class_id"] == cls_id
                ]

                if len(det_idx) == 0 or len(track_idx) == 0:
                    continue

                det_boxes = boxes[det_idx]
                track_boxes = np.vstack([
                    self.tracks[i]["box"] for i in track_idx
                ]).astype(np.float32)

                iou_matrix = self._iou_matrix(det_boxes, track_boxes)

                while iou_matrix.size > 0:
                    max_iou = float(iou_matrix.max())

                    if max_iou < self.iou_threshold:
                        break

                    d_local, t_local = np.unravel_index(
                        iou_matrix.argmax(),
                        iou_matrix.shape
                    )

                    d = int(det_idx[d_local])
                    t = int(track_idx[t_local])

                    if d in matched_det:
                        iou_matrix[d_local, :] = -1.0
                        continue

                    track = self.tracks[t]
                    track["box"] = boxes[d].copy()
                    track["score"] = float(scores[d])
                    track["missed"] = 0
                    track["hits"] += 1

                    matched_det.add(d)

                    iou_matrix[d_local, :] = -1.0
                    iou_matrix[:, t_local] = -1.0

        if len(boxes) > 0:
            for d in range(len(boxes)):
                if d in matched_det or scores[d] < self.min_new_score:
                    continue

                self.tracks.append(
                    {
                        "id": self.next_id,
                        "class_id": int(class_ids[d]),
                        "box": boxes[d].copy(),
                        "score": float(scores[d]),
                        "missed": 0,
                        "hits": 1,
                    }
                )

                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_age]

        out_boxes = []
        out_scores = []
        out_class_ids = []
        out_track_ids = []

        for track in self.tracks:
            if track["missed"] == 0:
                out_boxes.append(track["box"])
                out_scores.append(track["score"])
                out_class_ids.append(track["class_id"])
                out_track_ids.append(track["id"])

        if not out_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.asarray(out_boxes, dtype=np.float32),
            np.asarray(out_scores, dtype=np.float32),
            np.asarray(out_class_ids, dtype=np.int32),
            np.asarray(out_track_ids, dtype=np.int32),
        )


def empty_tracking_result():
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
    )


def get_boxmot_class(tracker_name: str):
    """
    Возвращает класс трекера из новой структуры boxmot 25.x.
    """
    if not HAS_BOXMOT:
        return None, None

    if tracker_name not in BOXMOT_TRACKER_PATHS:
        return None, None

    module_path, class_name = BOXMOT_TRACKER_PATHS[tracker_name]

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls, f"{module_path}.{class_name}"
    except Exception as e:
        print(f"❌ Не удалось загрузить {module_path}.{class_name}: {e}")
        return None, None


def build_boxmot_kwargs(tracker_name: str, TrackerClass, args, device):
    """
    Подбирает только те аргументы, которые поддерживает конкретный трекер.
    """
    sig = inspect.signature(TrackerClass.__init__)
    params = sig.parameters

    kwargs = {}

    high_thresh = float(args.box_threshold)
    low_thresh = max(0.10, high_thresh * 0.5)

    # Общие параметры
    if "track_buffer" in params:
        kwargs["track_buffer"] = int(args.max_age)

    if "device" in params:
        kwargs["device"] = device

    if "half" in params:
        kwargs["half"] = bool(args.fp16 and device.type == "cuda")

    if "reid_weights" in params and getattr(args, "reid_weights", None):
        kwargs["reid_weights"] = Path(args.reid_weights)

    # Индивидуальные параметры для разных трекеров
    if tracker_name == "botsort":
        if "track_high_thresh" in params:
            kwargs["track_high_thresh"] = high_thresh
        if "track_low_thresh" in params:
            kwargs["track_low_thresh"] = low_thresh
        if "new_track_thresh" in params:
            kwargs["new_track_thresh"] = max(high_thresh, 0.5)
        if "match_thresh" in params:
            kwargs["match_thresh"] = 0.8

    elif tracker_name == "bytetrack":
        if "track_thresh" in params:
            kwargs["track_thresh"] = high_thresh
        if "min_conf" in params:
            kwargs["min_conf"] = low_thresh
        if "match_thresh" in params:
            kwargs["match_thresh"] = 0.8

    elif tracker_name == "ocsort":
        if "min_conf" in params:
            kwargs["min_conf"] = low_thresh

    elif tracker_name == "deepocsort":
        if "min_conf" in params:
            kwargs["min_conf"] = low_thresh

    elif tracker_name == "strongsort":
        if "min_conf" in params:
            kwargs["min_conf"] = low_thresh

    elif tracker_name == "hybridsort":
        if "track_thresh" in params:
            kwargs["track_thresh"] = high_thresh
        if "low_thresh" in params:
            kwargs["low_thresh"] = low_thresh

    elif tracker_name == "occluboost":
        if "track_low_thresh" in params:
            kwargs["track_low_thresh"] = low_thresh
        if "new_track_thresh" in params:
            kwargs["new_track_thresh"] = max(high_thresh, 0.6)

    elif tracker_name == "boosttrack":
        # BoostTrack по умолчанию может работать без embeddings.
        # Если хочешь включить ReID для BoostTrack, раскомментируй строки ниже:
        #
        # if "use_embeddings" in params:
        #     kwargs["use_embeddings"] = True
        pass

    elif tracker_name == "sfsort":
        if "high_th" in params:
            kwargs["high_th"] = high_thresh
        if "low_th" in params:
            kwargs["low_th"] = low_thresh

    return kwargs


def create_tracker(args, device):
    tracker_name = args.tracker

    if tracker_name == "simple":
        candidates = []
    elif tracker_name == "auto":
        candidates = ["botsort", "bytetrack", "ocsort"]
    else:
        candidates = [tracker_name]

    if HAS_BOXMOT:
        for cand in candidates:
            TrackerClass, found_path = get_boxmot_class(cand)

            if TrackerClass is None:
                print(f"⚠️ Не найден класс для трекера '{cand}'.")
                continue

            try:
                kwargs = build_boxmot_kwargs(cand, TrackerClass, args, device)
                tracker = TrackerClass(**kwargs)
                print(f"✅ Успешно инициализирован {found_path}")
                return tracker, cand

            except Exception as e:
                print(f"⚠️ Ошибка инициализации {found_path} с аргументами: {e}")

                try:
                    tracker = TrackerClass()
                    print(f"✅ Успешно инициализирован {found_path} без аргументов.")
                    return tracker, cand
                except Exception as e2:
                    print(f"❌ Ошибка инициализации {found_path} без аргументов: {e2}")
    else:
        print("❌ boxmot недоступен:", BOXMOT_IMPORT_ERROR)

    # Fallback на supervision ByteTrack
    if tracker_name in ("bytetrack", "auto") and HAS_SUPERVISION and sv is not None:
        if hasattr(sv, "ByteTrack"):
            try:
                tracker = sv.ByteTrack(
                    track_activation_threshold=max(0.20, args.box_threshold * 0.8),
                    lost_track_buffer=args.max_age,
                )
                print("✅ Используется ByteTrack из supervision.")
                return tracker, "bytetrack_sv"
            except Exception as e:
                print("⚠️ Ошибка инициализации supervision ByteTrack:", e)
        else:
            print("⚠️ supervision установлен, но в нём нет ByteTrack.")
    elif tracker_name == "bytetrack" and not HAS_SUPERVISION:
        print("⚠️ supervision недоступен:", SV_IMPORT_ERROR)

    print("⚠️ Используется запасной SimpleIoUTracker.")

    min_new_score = max(0.15, args.box_threshold * 0.7)
    tracker = SimpleIoUTracker(
        max_age=args.max_age,
        iou_threshold=args.iou_threshold,
        min_new_score=min_new_score,
    )

    return tracker, "simple"


def convert_boxmot_outputs(outputs):
    """
    Приводит вывод трекера к формату:
    boxes, confidence, class_ids, track_ids
    """
    if outputs is None:
        return empty_tracking_result()

    # Если вернулся объект типа Detections/Tracks с полем xyxy
    if not isinstance(outputs, np.ndarray):
        if hasattr(outputs, "xyxy"):
            try:
                out_boxes = np.asarray(outputs.xyxy, dtype=np.float32)

                if out_boxes.ndim == 1:
                    if out_boxes.size == 0:
                        return empty_tracking_result()
                    if out_boxes.size % 4 == 0:
                        out_boxes = out_boxes.reshape(-1, 4)
                    else:
                        return empty_tracking_result()

                n = out_boxes.shape[0]

                track_ids = None
                for attr in ["ids", "tracker_id", "track_id"]:
                    if hasattr(outputs, attr):
                        track_ids = np.asarray(getattr(outputs, attr), dtype=np.int32)
                        break

                if track_ids is None or len(track_ids) != n:
                    track_ids = np.full(n, -1, dtype=np.int32)

                class_ids = None
                for attr in ["class_id", "cls", "classes"]:
                    if hasattr(outputs, attr):
                        class_ids = np.asarray(getattr(outputs, attr), dtype=np.int32)
                        break

                if class_ids is None or len(class_ids) != n:
                    class_ids = np.zeros(n, dtype=np.int32)

                confidence = None
                for attr in ["confidence", "conf", "scores"]:
                    if hasattr(outputs, attr):
                        confidence = np.asarray(getattr(outputs, attr), dtype=np.float32)
                        break

                if confidence is None or len(confidence) != n:
                    confidence = np.ones(n, dtype=np.float32)

                return out_boxes, confidence, class_ids, track_ids

            except Exception:
                pass

        # Если есть методы конвертации в numpy
        for method_name in ["to_numpy", "numpy", "as_numpy", "to_array"]:
            method = getattr(outputs, method_name, None)
            if callable(method):
                try:
                    outputs = method()
                    break
                except Exception:
                    pass

        if not isinstance(outputs, np.ndarray):
            try:
                outputs = np.asarray(outputs)
            except Exception:
                return empty_tracking_result()

    arr = outputs

    if arr.size == 0:
        return empty_tracking_result()

    if arr.ndim == 1:
        if arr.size >= 5:
            arr = arr.reshape(1, -1)
        else:
            return empty_tracking_result()

    if arr.ndim != 2 or arr.shape[1] < 5:
        print(f"Неожиданный формат вывода трекера: {arr.shape}")
        return empty_tracking_result()

    try:
        out_boxes = arr[:, :4].astype(np.float32)
        track_ids = arr[:, 4].astype(np.int32)
    except Exception:
        return empty_tracking_result()

    n = out_boxes.shape[0]

    def looks_like_class_column(x: np.ndarray) -> bool:
        if x.size == 0:
            return False

        if not np.all(np.isfinite(x)):
            return False

        integer_like = np.allclose(x, np.round(x), atol=1e-3, rtol=0.0)
        in_class_range = np.all((x >= -1) & (x < len(CLASS_NAMES)))

        return bool(integer_like and in_class_range)

    # Если колонок больше 5, пытаемся понять, где класс, а где confidence
    if arr.shape[1] >= 7:
        try:
            col5 = arr[:, 5].astype(np.float64)
            col6 = arr[:, 6].astype(np.float64)
        except Exception:
            col5 = np.zeros(n, dtype=np.float64)
            col6 = np.ones(n, dtype=np.float64)

        col5_is_class = looks_like_class_column(col5)
        col6_is_class = looks_like_class_column(col6)

        if col5_is_class and not col6_is_class:
            class_ids = col5
            confidence = col6
        elif col6_is_class and not col5_is_class:
            class_ids = col6
            confidence = col5
        elif col5_is_class and col6_is_class:
            # Неоднозначно, по умолчанию считаем 5-й столбец классом
            class_ids = col5
            confidence = col6
        else:
            # Явного класса нет, считаем 5-й столбец уверенностью
            class_ids = np.zeros(n, dtype=np.float64)
            confidence = col5

    elif arr.shape[1] == 6:
        try:
            col5 = arr[:, 5].astype(np.float64)
        except Exception:
            col5 = np.ones(n, dtype=np.float64)

        if looks_like_class_column(col5):
            class_ids = col5
            confidence = np.ones(n, dtype=np.float64)
        else:
            class_ids = np.zeros(n, dtype=np.float64)
            confidence = col5

    else:
        class_ids = np.zeros(n, dtype=np.float64)
        confidence = np.ones(n, dtype=np.float64)

    class_ids = np.where(np.isfinite(class_ids), class_ids, 0.0)
    confidence = np.where(np.isfinite(confidence), confidence, 1.0).astype(np.float32)

    return (
        out_boxes,
        confidence,
        np.round(class_ids).astype(np.int32),
        track_ids,
    )


def update_tracker(tracker, tracker_type, boxes, class_ids, scores, frame=None):
    # Трекеры из boxmot 25.x
    if tracker_type in BOXMOT_TRACKER_PATHS:
        if len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            # boxmot обычно принимает формат [x1, y1, x2, y2, conf, class_id]
            dets = np.hstack([
                boxes.astype(np.float32),
                scores[:, None].astype(np.float32),
                class_ids[:, None].astype(np.float32),
            ])

        try:
            img = frame if frame is not None else np.zeros((10, 10, 3), dtype=np.uint8)
            outputs = tracker.update(dets, img)
        except Exception as e:
            print(f"Ошибка трекера {tracker_type}: {e}")
            return empty_tracking_result()

        return convert_boxmot_outputs(outputs)

    # supervision ByteTrack fallback
    if tracker_type == "bytetrack_sv":
        if not HAS_SUPERVISION or sv is None:
            return empty_tracking_result()

        if len(boxes) == 0:
            detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
                confidence=np.empty((0,), dtype=np.float32),
            )
        else:
            detections = sv.Detections(
                xyxy=boxes.astype(np.float32),
                class_id=class_ids.astype(np.int32),
                confidence=scores.astype(np.float32),
            )

        try:
            tracked = tracker.update_with_detections(detections)
        except Exception:
            return empty_tracking_result()

        if tracked is None or len(tracked) == 0:
            return empty_tracking_result()

        if tracked.tracker_id is not None:
            track_ids = np.array(
                [-1 if tid is None else int(tid) for tid in tracked.tracker_id],
                dtype=np.int32,
            )
        else:
            track_ids = np.full(len(tracked), -1, dtype=np.int32)

        if tracked.class_id is not None:
            tracked_class_ids = np.asarray(tracked.class_id, dtype=np.int32)
        else:
            tracked_class_ids = np.zeros(len(tracked), dtype=np.int32)

        if tracked.confidence is not None:
            confidence = np.asarray(tracked.confidence, dtype=np.float32)
        else:
            confidence = np.ones(len(tracked), dtype=np.float32)

        return (
            np.asarray(tracked.xyxy, dtype=np.float32),
            confidence,
            tracked_class_ids,
            track_ids,
        )

    # SimpleIoUTracker
    return tracker.update(boxes, class_ids, scores, frame)


def draw_detections(frame, boxes, scores, class_ids, track_ids, fps=None):
    if track_ids is None:
        track_ids = [-1] * len(boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(np.int32)
        score = float(scores[i])
        cls_id = int(class_ids[i])
        track_id = int(track_ids[i]) if i < len(track_ids) else -1

        label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if track_id >= 0:
            text = f"{label}:{track_id} {score:.2f}"
        else:
            text = f"{label} {score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        top = max(0, y1 - th - baseline - 4)

        cv2.rectangle(frame, (x1, top), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def open_capture(source: str):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник: {source}")

    return cap


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = select_device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if HAS_BOXMOT:
        print("✅ boxmot доступен:", getattr(boxmot, "__version__", "unknown version"))
    else:
        print("❌ boxmot недоступен:", BOXMOT_IMPORT_ERROR)

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, device)

    tracker, tracker_type = create_tracker(args, device)
    print(f"Tracker selected: {tracker_type}")

    cap = open_capture(args.source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if fps <= 0:
        fps = 30.0

    output_fps = fps / max(1, args.frame_stride)

    writer = None

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать видеофайл: {args.output}")

    frame_idx = 0
    fps_ema = None
    display_enabled = not args.no_display

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame_idx % max(1, args.frame_stride) != 0:
            frame_idx += 1
            continue

        start_time = time.time()

        boxes, scores, class_ids = detect_objects(model, processor, frame, device, args)
        boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, args.nms_iou)

        boxes, scores, class_ids, track_ids = update_tracker(
            tracker,
            tracker_type,
            boxes,
            class_ids,
            scores,
            frame,
        )

        elapsed = time.time() - start_time
        instant_fps = 1.0 / max(elapsed, 1e-6)
        fps_ema = instant_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * instant_fps

        draw_detections(frame, boxes, scores, class_ids, track_ids, fps_ema)

        if writer is not None:
            writer.write(frame)

        if display_enabled:
            try:
                cv2.imshow("Grounding DINO tracking", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            except cv2.error:
                print("GUI недоступен. Продолжаю без окна.")
                display_enabled = False

        frame_idx += 1

    cap.release()

    if writer is not None:
        writer.release()

    if display_enabled:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

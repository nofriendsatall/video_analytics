import argparse
import os
import time
import inspect

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Классы, которые нам нужны.
# Grounding DINO лучше понимает английские текстовые запросы.
CLASS_NAMES = ["person", "table"]
CLASS_COLORS = {
    0: (0, 220, 0),    # person: зелёный
    1: (255, 160, 0),  # table: синий/оранжевый
}

# Текстовый запрос для Grounding DINO.
# Можно расширить синонимами, например: desk, dining table.
TEXT_PROMPT = "person . table . desk . dining table ."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Детекция людей и столов с использованием Grounding DINO."
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Путь к видео, RTSP-ссылка или индекс веб-камеры, например 0.",
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace модель Grounding DINO: tiny быстрее, base точнее.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство инференса.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        help="Порог уверенности для боксов.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Порог соответствия текстовому запросу.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="IoU-порог для NMS.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Максимальная сторона кадра для детекции. 0 - без уменьшения.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Путь для сохранения выходного видео, например out.mp4.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Не показывать окно с видео.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Использовать fp16 autocast на CUDA. Обычно быстрее на современных GPU.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Обрабатывать каждый N-й кадр. 1 - все кадры.",
    )

    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. Проверь драйвер, PyTorch и установку pytorch-cuda."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    return model, processor


def resize_for_inference(frame: np.ndarray, max_size: int):
    """
    Уменьшает кадр для детекции, но оригинал используется для отрисовки.
    Возвращает уменьшенный кадр и масштаб.
    """
    h, w = frame.shape[:2]

    if max_size is None or max_size <= 0:
        return frame, 1.0

    longest = max(h, w)
    if longest <= max_size:
        return frame, 1.0

    scale = max_size / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def decode_labels(labels, processor):
    """
    Приводит метки из результата модели к списку строк.
    """
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
        # Иногда возвращается [[...]] для батча из одного изображения.
        if len(labels) > 0 and isinstance(labels[0], (list, tuple)):
            return list(labels[0])
        return list(labels)

    return [labels]


def normalize_label(text: str) -> str:
    """
    Сводим разные варианты подписей к двум классам:
    - person
    - table
    """
    t = str(text).lower().strip()
    t = t.replace(".", " ").replace(",", " ").replace("_", " ").strip()

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


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
):
    """
    Class-aware NMS через torchvision.
    """
    if len(boxes) == 0 or iou_threshold <= 0:
        return boxes, scores, class_ids

    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    cls_t = torch.from_numpy(class_ids.astype(np.int64))

    # Смещаем боксы разных классов, чтобы NMS не смешивал классы.
    offset = cls_t * 1_000_000.0
    boxes_for_nms = boxes_t + offset.unsqueeze(1)

    keep = nms(boxes_for_nms, scores_t, iou_threshold)
    keep = keep.cpu().numpy()

    return boxes[keep], scores[keep], class_ids[keep]


def detect_objects(
    model,
    processor,
    frame_bgr: np.ndarray,
    device: torch.device,
    args,
):
    """
    Детекция людей и столов через Grounding DINO.
    Возвращает boxes xyxy в координатах исходного кадра.

    Эта версия совместима со старыми и новыми версиями transformers,
    где параметры постпроцессинга могут называться по-разному:
    - box_threshold / threshold / score_threshold
    """
    infer_img, scale = resize_for_inference(frame_bgr, args.max_size)
    h, w = infer_img.shape[:2]

    rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    inputs = processor(
        images=pil_image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    ).to(device)

    use_amp = args.fp16 and device.type == "cuda"

    if use_amp:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
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

    # target_sizes нужен, чтобы получить координаты в пикселях.
    if has_var_kwargs or "target_sizes" in pp_params:
        pp_kwargs["target_sizes"] = [(h, w)]
        used_target_sizes = True

    # В разных версиях transformers порог может называться по-разному.
    if "box_threshold" in pp_params:
        pp_kwargs["box_threshold"] = args.box_threshold
    elif "threshold" in pp_params:
        pp_kwargs["threshold"] = args.box_threshold
    elif "score_threshold" in pp_params:
        pp_kwargs["score_threshold"] = args.box_threshold
    elif has_var_kwargs:
        pp_kwargs["threshold"] = args.box_threshold

    # Текстовый порог тоже передаём только если он поддерживается.
    if "text_threshold" in pp_params or has_var_kwargs:
        pp_kwargs["text_threshold"] = args.text_threshold

    results = postprocess_fn(
        outputs,
        inputs["input_ids"],
        **pp_kwargs,
    )[0]

    boxes = results["boxes"].detach().float().cpu().numpy()
    scores = results["scores"].detach().float().cpu().numpy()

    # В новых версиях transformers рекомендуется использовать text_labels.
    if "text_labels" in results and results["text_labels"] is not None:
        labels_raw = results["text_labels"]
    else:
        labels_raw = results.get("labels", [])

    # Если версия не поддерживает target_sizes, координаты могут быть нормализованы.
    if not used_target_sizes:
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)

    labels = decode_labels(labels_raw, processor)

    # На случай, если модель вернула одну метку на несколько боксов.
    if len(labels) == 1 and len(boxes) > 1:
        labels = labels * len(boxes)

    n = min(len(boxes), len(scores), len(labels))
    boxes = boxes[:n]
    scores = scores[:n]
    labels = labels[:n]

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

    # Возвращаем координаты к исходному кадру.
    H, W = frame_bgr.shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    return (
        boxes[valid].astype(np.float32),
        scores[valid].astype(np.float32),
        class_ids[valid].astype(np.int32),
    )


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    fps: float = None,
):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(np.int32)
        score = float(scores[i])
        cls_id = int(class_ids[i])

        label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        top = max(0, y1 - th - baseline - 4)

        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + tw, y1),
            color,
            -1,
        )

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
    if source.isdigit():
        src = int(source)
    else:
        src = source

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source}")

    return cap


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = select_device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, device)

    cap = open_capture(args.source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    output_fps = fps / max(1, args.frame_stride)

    writer = None
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
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

        boxes, scores, class_ids = detect_objects(
            model=model,
            processor=processor,
            frame_bgr=frame,
            device=device,
            args=args,
        )

        boxes, scores, class_ids = apply_nms(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            iou_threshold=args.nms_iou,
        )

        elapsed = time.time() - start_time
        instant_fps = 1.0 / max(elapsed, 1e-6)

        if fps_ema is None:
            fps_ema = instant_fps
        else:
            fps_ema = 0.9 * fps_ema + 0.1 * instant_fps

        draw_detections(
            frame=frame,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            fps=fps_ema,
        )

        if writer is not None:
            writer.write(frame)

        if display_enabled:
            try:
                cv2.imshow("Grounding DINO detection: person + table", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            except cv2.error:
                print(
                    "GUI недоступен. Продолжаю обработку без отображения окна. "
                    "Используй --no-display, чтобы убрать эту ошибку."
                )
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
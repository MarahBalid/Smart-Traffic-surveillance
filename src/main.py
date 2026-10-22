from pathlib import Path
from collections import Counter, deque
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"

VIDEO_PATH = DATA_DIR / "traffic.mp4"

MODEL_CAR_BUS = MODELS_DIR / "base" / "yolo11n.pt"

MODEL_TRUCK_VAN = MODELS_DIR / "custom" / "weights (2).pt"

MODEL_MOTORCYCLES = MODELS_DIR / "vehicle2" / "weights.pt"

OUTPUT_PATH = OUTPUT_DIR / "merged_traffic.mp4"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.20
IOU_NMS = 0.50
IMGSZ = 640
MAX_DET_PER_MODEL = 100

# Canonical class IDs used by this program:
# 0=car, 1=bus, 2=truck, 3=van, 4=motorcycle, 5=police_motorcycle
CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "van",
    4: "motorcycle",
    5: "police_motorcycle",
}

# OpenCV colors are BGR.
VEHICLE_COLORS = {
    "car": (255, 0, 0),
    "bus": (0, 255, 0),
    "truck": (0, 0, 255),
    "van": (255, 0, 255),
    "motorcycle": (0, 255, 255),
    "police_motorcycle": (0, 165, 255),
}

TOLL_RATES = {
    "car": 1.0,
    "bus": 2.0,
    "truck": 2.5,
    "van": 1.5,
    "motorcycle": 0.5,
    "police_motorcycle": 0.5,
}

# The original example flags ordinary motorcycles as class violations.
# Change this set if your project's actual violation rule is different.
CLASS_VIOLATION_CLASSES = {"motorcycle"}

LINE_POSITION_RATIO = 0.70
LINE_OFFSET = 8

MAX_SPEED = 80.0
SPEED_FACTOR = 2.5

# Unified tracker settings. IDs are assigned only after merging detections.
TRACK_MAX_MISSED = 20
TRACK_MIN_IOU = 0.10
TRACK_MAX_CENTER_DISTANCE = 90.0
TRACK_CLASS_HISTORY = 7


def adaptive_preprocessing(frame):
    """Improve low-light contrast while leaving normally lit frames unchanged."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    if mean_brightness < 85:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return frame


def box_iou(box_a, box_b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def center_distance(box_a, box_b):
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def class_aware_nms(detections, iou_threshold=IOU_NMS):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept = []

    while detections:
        best = detections.pop(0)
        duplicate = False

        for existing in kept:
            iou = box_iou(best["box"], existing["box"])

            if iou >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(best)

    return kept


class UnifiedIoUTracker:
    """
    Lightweight unified tracker for detections merged from multiple models.
    It maintains one ID space for the entire merged detection list.
    """

    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.max_missed = TRACK_MAX_MISSED

    def update(self, detections):
        for track in self.tracks.values():
            track["missed"] += 1

        candidates = []
        for track_id, track in self.tracks.items():
            for det_index, det in enumerate(detections):
                if track["class_id"] != det["class_id"]:
                    continue

                iou = box_iou(track["box"], det["box"])
                distance = center_distance(track["box"], det["box"])

                if iou >= TRACK_MIN_IOU or distance <= TRACK_MAX_CENTER_DISTANCE:
                    # Prefer overlap; distance is a secondary matching signal.
                    score = iou - min(distance / 1000.0, 0.09)
                    candidates.append((score, track_id, det_index))

        candidates.sort(reverse=True, key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()
        assignments = {}

        for _, track_id, det_index in candidates:
            if track_id in matched_tracks or det_index in matched_detections:
                continue

            matched_tracks.add(track_id)
            matched_detections.add(det_index)
            assignments[det_index] = track_id

        for det_index, det in enumerate(detections):
            if det_index in assignments:
                track_id = assignments[det_index]
                track = self.tracks[track_id]
                track["box"] = det["box"]
                track["confidence"] = det["confidence"]
                track["missed"] = 0
                track["class_history"].append(det["class_id"])
                track["class_id"] = Counter(track["class_history"]).most_common(1)[0][0]
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "box": det["box"],
                    "class_id": det["class_id"],
                    "confidence": det["confidence"],
                    "missed": 0,
                    "class_history": deque(
                        [det["class_id"]], maxlen=TRACK_CLASS_HISTORY
                    ),
                    "previous_center": None,
                    "counted": False,
                }
                assignments[det_index] = track_id

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]
        for track_id in expired:
            del self.tracks[track_id]

        output = []
        for det_index, det in enumerate(detections):
            track_id = assignments[det_index]
            track = self.tracks[track_id]
            output.append(
                {
                    **det,
                    "track_id": track_id,
                    "class_id": track["class_id"],
                    "box": track["box"],
                    "previous_center": track["previous_center"],
                }
            )

        return output


def load_models():
    required_paths = [
        MODEL_TRUCK_VAN,
        MODEL_MOTORCYCLES,
        MODEL_CAR_BUS,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    print("Loading only the three approved models...")
    model_truck_van = YOLO(str(MODEL_TRUCK_VAN))
    model_motorcycles = YOLO(str(MODEL_MOTORCYCLES))
    model_car_bus = YOLO(str(MODEL_CAR_BUS))

   
    print("Truck/van model classes:", model_truck_van.names)
    print("Motorcycle model classes:", model_motorcycles.names)
    print("Base model classes:", model_car_bus.names)

    return model_truck_van, model_motorcycles, model_car_bus


def predict_model(model, frame, allowed_source_ids, canonical_mapping):
    """Run one model, retaining only explicitly approved source class IDs."""
    result = model.predict(
        source=frame,
        conf=CONFIDENCE,
        iou=IOU_NMS,
        imgsz=IMGSZ,
        max_det=MAX_DET_PER_MODEL,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    for box, source_class_id, confidence in zip(xyxy, class_ids, confidences):
        if source_class_id not in allowed_source_ids:
            continue

        canonical_id = canonical_mapping.get(source_class_id)
        if canonical_id is None:
            continue

        detections.append(
            {
                "box": tuple(float(v) for v in box),
                "class_id": canonical_id,
                "confidence": float(confidence),
            }
        )

    return detections


def detect_and_merge(frame, models):
    """
    Model 1: only Truck and Van.
    Model 2: only Motorcycle and Police Motorcycle.
    Base YOLO11n: only Car and Bus.
    """
    model_truck_van, model_motorcycles, model_car_bus = models

    truck_van = predict_model(
        model_truck_van,
        frame,
        allowed_source_ids={2, 3},
        canonical_mapping={2: 2, 3: 3},
    )

    motorcycles = predict_model(
        model_motorcycles,
        frame,
        allowed_source_ids={2, 3},
        canonical_mapping={2: 4, 3: 5},
    )

    car_bus = predict_model(
        model_car_bus,
        frame,
        allowed_source_ids={2, 5},
        canonical_mapping={2: 0, 5: 1},
    )

    # The three models have disjoint class responsibilities. NMS is
    # class-aware so adjacent vehicles of different classes are preserved.
    return class_aware_nms(truck_van + motorcycles + car_bus, IOU_NMS)


def draw_label(frame, box, text, color):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_top = max(0, y1 - text_h - baseline - 8)
    label_right = min(width - 1, x1 + text_w + 8)

    cv2.rectangle(
        frame,
        (x1, label_top),
        (label_right, max(label_top + 1, y1)),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 4, max(text_h + 2, y1 - baseline - 4)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    print("Used device:", DEVICE)
    print("Input video:", VIDEO_PATH)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}. "
            "Set VIDEO_PATH near the top of main.py to your video path."
        )

    models = load_models()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0

    delay = max(1, int(1000 / source_fps))
    line_y = int(height * LINE_POSITION_RATIO)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    tracker = UnifiedIoUTracker()
    counted_ids = set()
    track_history = {}
    track_previous_pos = {}
    total_revenue = 0.0
    total_counts = Counter()

    fps_smooth = 0.0
    prev_frame_time = time.time()

    print("Starting merged traffic detection, unified tracking, tolls and violations.")
    print("Press q in the video window to stop.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("End of video or frame read error.")
                break

            processed_frame = adaptive_preprocessing(frame)

            detections = detect_and_merge(processed_frame, models)
            tracked_detections = tracker.update(detections)

            # Keep previous center positions for speed estimation and line crossing.
            for item in tracked_detections:
                track_id = item["track_id"]
                class_id = item["class_id"]
                class_name = CLASS_NAMES[class_id]
                x1, y1, x2, y2 = item["box"]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=TRACK_CLASS_HISTORY)
                track_history[track_id].append(class_name)
                stable_class = Counter(track_history[track_id]).most_common(1)[0][0]

                previous_position = track_previous_pos.get(track_id)
                if previous_position is not None:
                    prev_x, prev_y = previous_position
                    pixel_distance = float(
                        np.hypot(center_x - prev_x, center_y - prev_y)
                    )
                    estimated_speed = pixel_distance * SPEED_FACTOR
                else:
                    prev_y = None
                    estimated_speed = 0.0

                track_previous_pos[track_id] = (center_x, center_y)

                is_speed_violation = estimated_speed > MAX_SPEED
                is_class_violation = stable_class in CLASS_VIOLATION_CLASSES
                is_violation = is_speed_violation or is_class_violation

                color = (
                    (0, 0, 255)
                    if is_violation
                    else VEHICLE_COLORS.get(stable_class, (255, 255, 255))
                )

                status_text = (
                    f"VIOLATION ID:{track_id} {stable_class} "
                    f"{estimated_speed:.1f} px-step"
                    if is_violation
                    else f"ID:{track_id} | {stable_class} "
                    f"{item['confidence']:.2f} | {estimated_speed:.1f} px-step"
                )

                # Count once per track ID when the vehicle crosses the line.
                crossed_line = False
                if prev_y is not None:
                    crossed_line = (prev_y < line_y <= center_y) or (
                        prev_y > line_y >= center_y
                    )

                inside_line_band = abs(center_y - line_y) <= LINE_OFFSET
                if track_id not in counted_ids and (crossed_line or inside_line_band):
                    counted_ids.add(track_id)
                    fee = TOLL_RATES.get(stable_class, 0.0)
                    total_revenue += fee
                    total_counts[stable_class] += 1
                    print(
                        f"Crossed | ID:{track_id} | Class:{stable_class} "
                        f"| Fee:${fee:.2f} | Total:${total_revenue:.2f}"
                    )

                draw_label(frame, item["box"], status_text, color)

            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)

            count_text = " | ".join(
                f"{CLASS_NAMES[class_id]}: {total_counts[CLASS_NAMES[class_id]]}"
                for class_id in CLASS_NAMES
            )
            cv2.rectangle(frame, (15, 10), (min(width - 15, 780), 105), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Revenue: ${total_revenue:.2f}",
                (30, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                count_text[:95],
                (30, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            now = time.time()
            elapsed = now - prev_frame_time
            prev_frame_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps_smooth = (
                    0.9 * fps_smooth + 0.1 * current_fps
                    if fps_smooth > 0
                    else current_fps
                )

            cv2.rectangle(
                frame,
                (max(0, width - 190), 15),
                (max(0, width - 20), 60),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                f"FPS: {int(fps_smooth)}",
                (max(0, width - 180), 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            display = cv2.resize(frame, (960, 540))
            # cv2.imshow("Smart Traffic - Three-Model Fusion", display)

            # key = cv2.waitKey(delay) & 0xFF
            # if key == ord("q"):
            #     break

    finally:
        cap.release()
        writer.release()
        # cv2.destroyAllWindows()

    print("\nProcessing finished.")
    print("Output video:", OUTPUT_PATH)
    print(f"Total revenue: ${total_revenue:.2f}")
    print("Final counts:", dict(total_counts))


if __name__ == "__main__":
    main()

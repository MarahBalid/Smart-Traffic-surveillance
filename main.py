from pathlib import Path
from collections import Counter, deque
import time
import os

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

CONFIDENCE = 0.25
IOU_NMS = 0.45
IMGSZ = 640
MAX_DET_PER_MODEL = 50

SKIP_FRAMES = 2

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "van",
    4: "motorcycle",
    5: "police_motorcycle",
}

VEHICLE_COLORS = {
    "car": (255, 0, 0),  # أزرق
    "bus": (0, 255, 0),  # أخضر
    "truck": (42, 42, 165),  # بني
    "van": (255, 0, 255),  # زهري
    "motorcycle": (150, 255, 255),  # أصفر فاتح
    "police_motorcycle": (0, 255, 128),  # أخضر فوسفوري
}

TOLL_RATES = {
    "car": 1.0,
    "bus": 2.0,
    "truck": 2.5,
    "van": 1.5,
    "motorcycle": 0.5,
    "police_motorcycle": 0.5,
}

CLASS_VIOLATION_CLASSES = {"motorcycle"}

LINE_POSITION_RATIO = 0.70
LINE_OFFSET = 20  # نطاق موسع لمنع تفويت أي مركبة أثناء العبور
LINE_COLOR = (255, 255, 255)

METERS_PER_PIXEL = 0.08
MAX_SPEED = 120.0

TRACK_MAX_MISSED = 20
TRACK_MIN_IOU = 0.10
TRACK_MAX_CENTER_DISTANCE = 90.0
TRACK_CLASS_HISTORY = 7

VIOLATION_DISPLAY_FRAMES = 20  # مدة ظهور التنبيه المرئي للمخالفة


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(by2, by2)

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
    required_paths = [MODEL_TRUCK_VAN, MODEL_MOTORCYCLES, MODEL_CAR_BUS]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    model_truck_van = YOLO(str(MODEL_TRUCK_VAN))
    model_motorcycles = YOLO(str(MODEL_MOTORCYCLES))
    model_car_bus = YOLO(str(MODEL_CAR_BUS))

    return model_truck_van, model_motorcycles, model_car_bus


@torch.inference_mode()
def predict_model(model, frame, allowed_source_ids, canonical_mapping):
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

    return class_aware_nms(truck_van + motorcycles + car_bus, IOU_NMS)


def draw_label(frame, box, text, color, is_violation=False, violation_type=""):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    box_color = (0, 0, 255) if is_violation else color
    box_thickness = 3 if is_violation else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    if is_violation:
        if violation_type == "speed":
            alert_banner = "🚨 SPEED VIOLATION!"
        elif violation_type == "motorcycle":
            alert_banner = "🚨 PROHIBITED MOTORCYCLE!"
        else:
            alert_banner = "🚨 POLICE ALERT!"

        (alert_w, alert_h), alert_base = cv2.getTextSize(alert_banner, font, 0.55, 2)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - alert_h - alert_base - 16)),
            (max(x1 + alert_w + 10, x2), y1),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            alert_banner,
            (x1 + 5, max(12, y1 - alert_base - 8)),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    offset_y = y1 - 30 if is_violation else y1
    label_top = max(0, offset_y - text_h - baseline - 8)
    label_right = min(width - 1, x1 + text_w + 8)

    cv2.rectangle(
        frame,
        (x1, label_top),
        (label_right, max(label_top + 1, offset_y)),
        box_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 4, max(text_h + 2, offset_y - baseline - 4)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_dashboards(
    frame,
    total_revenue,
    total_counts,
    current_fps,
    speed_violations_count,
    motorcycle_violations_count,
):
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 65), (20, 20, 20), -1)

    # لوحة شرف المخالفات في أعلى اليمين
    cv2.rectangle(overlay, (width - 240, 75), (width - 10, 155), (0, 0, 139), -1)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    revenue_text = f"Revenue: ${total_revenue:.2f}"
    cv2.putText(frame, revenue_text, (15, 25), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    counts_list = []
    for c_id, c_name in CLASS_NAMES.items():
        counts_list.append(f"{c_name}: {total_counts[c_name]}")
    counts_str = " | ".join(counts_list)
    cv2.putText(
        frame, counts_str, (15, 48), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )

    fps_text = f"FPS: {int(current_fps)}"
    (fps_w, _), _ = cv2.getTextSize(fps_text, font, 0.6, 2)
    cv2.putText(
        frame,
        fps_text,
        (width - fps_w - 20, 35),
        font,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "🚨 VIOLATIONS SUMMARY",
        (width - 230, 95),
        font,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"• Speed Violations: {speed_violations_count}",
        (width - 230, 120),
        font,
        0.42,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"• Moto Violations:  {motorcycle_violations_count}",
        (width - 230, 142),
        font,
        0.42,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    models = load_models()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 25.0

    line_y = int(height * LINE_POSITION_RATIO)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (width, height),
    )

    tracker = UnifiedIoUTracker()
    counted_ids = set()
    notified_violation_ids = set()
    track_history = {}
    track_previous_pos = {}
    track_speed_history = {}

    violation_timer = {}
    vehicle_violation_type = {}

    speed_violations_count = 0
    motorcycle_violations_count = 0

    total_revenue = 0.0
    total_counts = Counter({c_name: 0 for c_name in CLASS_NAMES.values()})

    frame_count = 0
    last_detections = []

    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            frame_count += 1

            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            prev_time = curr_time
            current_processing_fps = (
                1.0 / elapsed_time if elapsed_time > 0 else video_fps
            )

            if frame_count % SKIP_FRAMES == 0 or not last_detections:
                last_detections = detect_and_merge(frame, models)

            tracked_detections = tracker.update(last_detections)

            for item in tracked_detections:
                track_id = item["track_id"]
                class_id = item["class_id"]

                if isinstance(class_id, int):
                    class_name = CLASS_NAMES.get(class_id, "car")
                else:
                    class_name = str(class_id)

                x1, y1, x2, y2 = item["box"]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=TRACK_CLASS_HISTORY)
                track_history[track_id].append(class_name)

                stable_class = Counter(track_history[track_id]).most_common(1)[0][0]

                if center_x < (width / 2):
                    direction_text = "Left"
                else:
                    direction_text = "Right"

                previous_position = track_previous_pos.get(track_id)
                if previous_position is not None:
                    prev_x, prev_y = previous_position
                    pixel_distance = float(
                        np.hypot(center_x - prev_x, center_y - prev_y)
                    )
                    raw_speed_kmh = (
                        pixel_distance
                        * (video_fps / SKIP_FRAMES)
                        * METERS_PER_PIXEL
                        * 3.6
                    )
                else:
                    prev_y = None
                    raw_speed_kmh = 0.0

                track_previous_pos[track_id] = (center_x, center_y)

                if track_id not in track_speed_history:
                    track_speed_history[track_id] = deque(maxlen=5)
                track_speed_history[track_id].append(raw_speed_kmh)
                estimated_speed_kmh = float(np.mean(track_speed_history[track_id]))

                # 🚀 الحل الجذري للخطأ: الفحص المباشر للنصوص داخل track_history دون استخدام القاموس بمفاتيح نصية
                is_police_track = stable_class == "police_motorcycle" or any(
                    name == "police_motorcycle" for name in track_history[track_id]
                )

                near_checkpoint = center_y >= (line_y - 120)

                is_speed_violation = (
                    estimated_speed_kmh > MAX_SPEED
                ) and near_checkpoint
                is_class_violation = (
                    stable_class in CLASS_VIOLATION_CLASSES and not is_police_track
                ) and near_checkpoint

                is_violation_now = is_speed_violation or is_class_violation

                if is_violation_now:
                    violation_timer[track_id] = VIOLATION_DISPLAY_FRAMES
                    if is_speed_violation:
                        vehicle_violation_type[track_id] = "speed"
                    elif is_class_violation:
                        vehicle_violation_type[track_id] = "motorcycle"

                    if track_id not in notified_violation_ids:
                        notified_violation_ids.add(track_id)
                        if is_speed_violation:
                            speed_violations_count += 1
                            print(
                                f"[Police Alert] ⚡ Speed Violation - Side: {direction_text} | ID:{track_id} | Speed: {estimated_speed_kmh:.1f} km/h"
                            )
                        elif is_class_violation:
                            motorcycle_violations_count += 1
                            print(
                                f"[Police Alert] 🏍️ Prohibited Motorcycle Violation - Side: {direction_text} | ID:{track_id}"
                            )

                show_active_violation = False
                v_type = ""
                if track_id in violation_timer and violation_timer[track_id] > 0:
                    show_active_violation = True
                    v_type = vehicle_violation_type.get(track_id, "")
                    violation_timer[track_id] -= 1

                color = VEHICLE_COLORS.get(stable_class, (255, 255, 255))

                status_text = f"ID:{track_id} | {stable_class} | {direction_text} | {estimated_speed_kmh:.1f} km/h"

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

                draw_label(
                    frame,
                    item["box"],
                    status_text,
                    color,
                    is_violation=show_active_violation,
                    violation_type=v_type,
                )

            cv2.line(frame, (0, line_y), (width, line_y), LINE_COLOR, 3)

            draw_dashboards(
                frame,
                total_revenue,
                total_counts,
                current_processing_fps,
                speed_violations_count,
                motorcycle_violations_count,
            )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    report_path = PROJECT_ROOT / "outputs" / "reports" / "summary_report.txt"
    os.makedirs(PROJECT_ROOT / "outputs" / "reports", exist_ok=True)

    total_vehicles_crossed = len(counted_ids)
    total_violations_count = len(notified_violation_ids)
    vehicle_counts = total_counts

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== 🚗 Traffic Surveillance Final Report ==========\n")
        f.write(f"Total Vehicles Crossed: {total_vehicles_crossed}\n")
        f.write(f"Total Toll Revenue: ${total_revenue:.2f}\n")
        f.write(f"Total Violations Detected: {total_violations_count}\n")
        f.write(f" - Speed Violations: {speed_violations_count}\n")
        f.write(f" - Motorcycle Violations: {motorcycle_violations_count}\n")
        f.write("========================================================\n\n")

        f.write("📊 Vehicle Count Breakdown:\n")
        for vehicle_class, count in vehicle_counts.items():
            f.write(f" - {vehicle_class}: {count} vehicles\n")

    print(f"✅ Report successfully saved at: {report_path}")
    print("\nProcessing finished.")
    print("Output video:", OUTPUT_PATH)
    print(f"Total revenue: ${total_revenue:.2f}")
    print("Final counts:", dict(total_counts))


if __name__ == "__main__":
    main()

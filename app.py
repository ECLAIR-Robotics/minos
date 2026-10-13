import json
import math
import os
import time
from collections import deque
import cv2
import numpy as np


def load_camera_parameters(profile_name=None, frame_width=1280):
    config_file = 'camera_config.json'
    focal_length = None
    fire_width = 3.0

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            fire_width = config.get("default_fire_width_cm", 3.0)
            profiles = config.get("profiles", {})

            if profile_name in profiles:
                focal_length = profiles[profile_name]["focal_length"]
                fire_width = profiles[profile_name].get("default_fire_width_cm", fire_width)
                print(f"[CONFIG] Loaded saved profile: '{profile_name}' (Focal Length: {focal_length})")
            else:
                default_fov = config.get("default_fov_deg", 70.0)
                fov_rad = math.radians(default_fov)
                focal_length = frame_width / (2.0 * math.tan(fov_rad / 2.0))
                print(f"[CONFIG] Profile '{profile_name}' not found. Auto-estimated Focal Length: {focal_length:.2f}")

        except Exception as e:
            print(f"[WARN] Error reading config: {e}. Switching to FOV auto-estimation.")

    if focal_length is None:
        fov_rad = math.radians(70.0)
        focal_length = frame_width / (2.0 * math.tan(fov_rad / 2.0))

    return focal_length, fire_width


# --- CAMERA & SETUP ---
cap = cv2.VideoCapture(1)

# FORCE HIGH RESOLUTION & RE-ENABLE AUTOFOCUS SO IT STAYS SHARP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 enables autofocus so the camera stays sharp

window_name = 'Fire Detection with Distance Estimation'
cv2.namedWindow(window_name)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
SELECTED_PROFILE = "asus_zenbook"

FOCAL_LENGTH, KNOWN_FIRE_WIDTH = load_camera_parameters(
    profile_name=SELECTED_PROFILE, 
    frame_width=frame_width
)

ACTIVATION_THRESHOLD_SEC = 0.5
MAX_DISAPPEARED_SEC = 0.8
FLICKER_HISTORY_LEN = 8

frame_buffer = deque(maxlen=FLICKER_HISTORY_LEN)
tracked_flames = {}
next_flame_id = 0

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)


def calculate_distance(pixel_width, focal_length, known_width):
    if pixel_width <= 0:
        return 0.0
    return (known_width * focal_length) / pixel_width


def calculate_confidence(color_ratio, variance_val, aspect_ratio):
    color_score = min(color_ratio * 250, 40)
    flicker_score = min(variance_val * 2.0, 40)
    shape_score = 20 if 0.25 <= aspect_ratio <= 2.5 else 5
    return int(min(color_score + flicker_score + shape_score, 100))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray_frame.astype(np.float32))

    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    rgb_flame = (r > 210) & (g > 110) & (b < 130) & ((r.astype(int) - g.astype(int)) > 40) & (g > b)
    hsv_flame = ((h <= 10) | (h >= 170)) & (s >= 160) & (v >= 210)
    color_mask = np.uint8(rgb_flame & hsv_flame) * 255

    base_color_mask = np.uint8((r > 230) & (g > 200) & (b > 150) & (v >= 230)) * 255

    fg_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)
    motion_dilated = cv2.dilate(motion_mask, np.ones((7, 7), np.uint8), iterations=2)

    flicker_mask = np.zeros_like(gray_frame, dtype=np.uint8)
    std_dev_map = np.zeros_like(gray_frame, dtype=np.float32)

    if len(frame_buffer) == FLICKER_HISTORY_LEN:
        std_dev_map = np.std(np.array(frame_buffer), axis=0)
        flicker_pixels = (std_dev_map >= 18.0) & (std_dev_map <= 100.0)
        flicker_mask = np.uint8(flicker_pixels) * 255

    combined_mask = cv2.bitwise_and(color_mask, motion_dilated)
    combined_mask = cv2.bitwise_and(combined_mask, flicker_mask)

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_expand)
    combined_mask = cv2.dilate(combined_mask, kernel_expand, iterations=2)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 350:
            hull = cv2.convexHull(cnt)
            x, y, w, h = cv2.boundingRect(hull)

            if w < frame.shape[1] * 0.75 and h < frame.shape[0] * 0.75:
                roi_mask = combined_mask[y:y+h, x:x+w]
                roi_std = std_dev_map[y:y+h, x:x+w]
                roi_base = base_color_mask[y:y+h, x:x+w]

                color_ratio = np.count_nonzero(roi_mask) / (w * h)
                avg_flicker = np.mean(roi_std) if roi_std.size > 0 else 0
                aspect_ratio = float(w) / h if h > 0 else 1.0

                confidence = calculate_confidence(color_ratio, avg_flicker, aspect_ratio)
                distance_cm = calculate_distance(w, FOCAL_LENGTH, KNOWN_FIRE_WIDTH)

                base_points = np.argwhere(roi_base > 0)
                if len(base_points) > 0:
                    base_y_local, base_x_local = base_points[np.argmax(base_points[:, 0])]
                    base_coords = (x + int(base_x_local), y + int(base_y_local))
                else:
                    base_coords = (x + int(w / 2), y + h)

                cx = int(x + w / 2)
                cy = int(y + h / 2)

                current_detections.append({
                    'box': (x, y, w, h),
                    'center': (cx, cy),
                    'confidence': confidence,
                    'distance': distance_cm,
                    'base': base_coords
                })

    matched_ids = set()
    for det in current_detections:
        cx, cy = det['center']
        matched_id = None
        min_dist = 100

        for fid, flame_data in tracked_flames.items():
            if fid in matched_ids:
                continue
            tcx, tcy = flame_data['center']
            dist = math.hypot(cx - tcx, cy - tcy)
            if dist < min_dist:
                min_dist = dist
                matched_id = fid

        if matched_id is not None:
            tracked_flames[matched_id]['center'] = (cx, cy)
            tracked_flames[matched_id]['box'] = det['box']
            tracked_flames[matched_id]['confidence'] = det['confidence']
            tracked_flames[matched_id]['distance'] = det['distance']
            tracked_flames[matched_id]['base'] = det['base']
            tracked_flames[matched_id]['last_seen'] = current_time
            matched_ids.add(matched_id)
        else:
            tracked_flames[next_flame_id] = {
                'center': (cx, cy),
                'box': det['box'],
                'confidence': det['confidence'],
                'distance': det['distance'],
                'base': det['base'],
                'first_seen': current_time,
                'last_seen': current_time
            }
            matched_ids.add(next_flame_id)
            next_flame_id += 1

    for fid, flame_info in list(tracked_flames.items()):
        if current_time - flame_info['last_seen'] <= MAX_DISAPPEARED_SEC:
            x, y, w, h = flame_info['box']
            conf = flame_info['confidence']
            dist = flame_info['distance']
            base_x, base_y = flame_info['base']
            active_duration = current_time - flame_info['first_seen']

            if active_duration >= ACTIVATION_THRESHOLD_SEC and conf >= 50:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                label_status = f"FIRE {conf}% ({active_duration:.1f}s)"
                label_dist = f"Dist: {dist:.1f} cm ({dist/100:.2f} m)"
                
                cv2.putText(frame, label_status, (x, max(y - 25, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.putText(frame, label_dist, (x, max(y - 5, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.circle(frame, (base_x, base_y), 6, (255, 255, 0), -1)
                cv2.putText(frame, "BASE", (base_x + 8, base_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    stale_ids = [fid for fid, data in tracked_flames.items()
                 if current_time - data['last_seen'] > MAX_DISAPPEARED_SEC]
    for fid in stale_ids:
        del tracked_flames[fid]

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time
import math
from collections import deque

cap = cv2.VideoCapture(0)
window_name = 'Strict Flame Detection (No Yellow Noise)'
cv2.namedWindow(window_name)

# --- CONFIGURATION SETTINGS ---
ACTIVATION_THRESHOLD_SEC = 0.5  # Quick verification
MAX_DISAPPEARED_SEC = 0.8
FLICKER_HISTORY_LEN = 8

frame_buffer = deque(maxlen=FLICKER_HISTORY_LEN)
tracked_flames = {}
next_flame_id = 0

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)

def calculate_confidence(color_ratio, variance_val, aspect_ratio):
    """Calculates a strict confidence score (0 - 100%)."""
    color_score = min(color_ratio * 250, 40)       # Up to 40%
    flicker_score = min(variance_val * 2.0, 40)    # Up to 40%
    shape_score = 20 if 0.25 <= aspect_ratio <= 2.5 else 5  # Up to 20%
    return int(min(color_score + flicker_score + shape_score, 100))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray_frame.astype(np.float32))

    # 1. Denoise and prepare color spaces
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 2. STRICT FLAME COLOR RULES (Filters out Yellow, Skin, and Lights)
    # RGB Rule: Red dominates Green (R - G > 40), Green > Blue, and Red is high intensity (> 210)
    rgb_flame = (r > 210) & (g > 110) & (b < 130) & ((r.astype(int) - g.astype(int)) > 40) & (g > b)

    # HSV Rule: Hue limited STRICTLY to Red-Orange (Hue <= 10 OR Hue >= 170). Excludes Yellow (20-35)!
    # High Saturation (S >= 160) and Max Value (V >= 210)
    hsv_flame = ((h <= 10) | (h >= 170)) & (s >= 160) & (v >= 210)

    color_mask = np.uint8(rgb_flame & hsv_flame) * 255

    # 3. Base Core Detection (White-Hot / Blue-Yellow Core at root)
    base_color_mask = np.uint8((r > 230) & (g > 200) & (b > 150) & (v >= 230)) * 255

    # 4. Motion & Temporal Flicker Variance
    fg_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)
    motion_dilated = cv2.dilate(motion_mask, np.ones((7, 7), np.uint8), iterations=2)

    flicker_mask = np.zeros_like(gray_frame, dtype=np.uint8)
    std_dev_map = np.zeros_like(gray_frame, dtype=np.float32)

    if len(frame_buffer) == FLICKER_HISTORY_LEN:
        std_dev_map = np.std(np.array(frame_buffer), axis=0)
        # Real flame pixel variance is strictly above 18.0 (filters out webcam noise/reflections)
        flicker_pixels = (std_dev_map >= 18.0) & (std_dev_map <= 100.0)
        flicker_mask = np.uint8(flicker_pixels) * 255

    # 5. Combined Mask (Color AND Motion AND High Variance Flicker)
    combined_mask = cv2.bitwise_and(color_mask, motion_dilated)
    combined_mask = cv2.bitwise_and(combined_mask, flicker_mask)

    # 6. Morphological Expansion to Group Local Flame Segments
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_expand)
    combined_mask = cv2.dilate(combined_mask, kernel_expand, iterations=2)

    # 7. Contour Extraction
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 350:  # Ignore tiny noise artifacts
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

                # Locate base (bottom-most region with core heat)
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
                    'base': base_coords
                })

    # 8. Tracker Matching
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
            tracked_flames[matched_id]['base'] = det['base']
            tracked_flames[matched_id]['last_seen'] = current_time
            matched_ids.add(matched_id)
        else:
            tracked_flames[next_flame_id] = {
                'center': (cx, cy),
                'box': det['box'],
                'confidence': det['confidence'],
                'base': det['base'],
                'first_seen': current_time,
                'last_seen': current_time
            }
            matched_ids.add(next_flame_id)
            next_flame_id += 1

    # 9. Render Bounding Boxes
    for fid, flame_info in list(tracked_flames.items()):
        if current_time - flame_info['last_seen'] <= MAX_DISAPPEARED_SEC:
            x, y, w, h = flame_info['box']
            conf = flame_info['confidence']
            base_x, base_y = flame_info['base']
            active_duration = current_time - flame_info['first_seen']

            # Must meet duration threshold AND confidence >= 50%
            if active_duration >= ACTIVATION_THRESHOLD_SEC and conf >= 50:
                # Active Fire Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                label = f"FIRE {conf}% ({active_duration:.1f}s)"
                cv2.putText(frame, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw Base Indicator
                cv2.circle(frame, (base_x, base_y), 6, (255, 255, 0), -1)
                cv2.putText(frame, "BASE", (base_x + 8, base_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                # Verifying State
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Verifying {conf}% ({active_duration:.1f}s)", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 10. Clear Stale Trackers
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
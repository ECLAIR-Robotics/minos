import cv2
import numpy as np
import time
import math

cap = cv2.VideoCapture(0)
window_name = 'Unified Fire Detection'
cv2.namedWindow(window_name)

# Configurable activation threshold in seconds
ACTIVATION_THRESHOLD_SEC = 2.0
MAX_DISAPPEARED_SEC = 1.2  # Grace period so flickering doesn't reset the timer

# Tracking storage: { flame_id: {'center': (x, y), 'box': (x, y, w, h), 'first_seen': timestamp, 'last_seen': timestamp} }
tracked_flames = {}
next_flame_id = 0

# Background Subtractor with longer history to smooth out quick flicker drops
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=30, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 1. Blur frame to smooth sharp flame edges
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    
    # 2. Flame Color Range (RGB + HSV)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    rgb_flame = (r > 180) & (g > 100) & (b < 150) & (r > g) & (g > b)
    hsv_flame = (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 170)
    color_mask = np.uint8(rgb_flame & hsv_flame) * 255

    # 3. Motion Flicker Mask
    fg_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    # 4. Combine Color with Motion Buffer
    # We heavily dilate motion so short flicker pauses don't mask out the fire color
    motion_dilated = cv2.dilate(motion_mask, np.ones((15, 15), np.uint8), iterations=2)
    combined_mask = cv2.bitwise_and(color_mask, motion_dilated)

    # 5. Heavy Dilation to Merge Flame Fragments into ONE Big Box
    # A large kernel glues nearby split flame specks together into a single region
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, merge_kernel)
    combined_mask = cv2.dilate(combined_mask, merge_kernel, iterations=2)

    # 6. Extract Contours & Form Full Bounding Boxes via Convex Hull
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_detections = []

    if contours:
        # Group close contours into a unified convex hull if multiple components exist
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
        
        if valid_contours:
            all_points = np.vstack([cnt for cnt in valid_contours])
            hull = cv2.convexHull(all_points)
            x, y, w, h = cv2.boundingRect(hull)
            
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            current_detections.append({'box': (x, y, w, h), 'center': (cx, cy)})

    # 7. Match Detections to Active Flame Trackers
    matched_ids = set()
    
    for det in current_detections:
        cx, cy = det['center']
        matched_id = None
        min_dist = 150  # Larger radius to follow flame movement without losing track

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
            tracked_flames[matched_id]['last_seen'] = current_time
            matched_ids.add(matched_id)
            target_id = matched_id
        else:
            tracked_flames[next_flame_id] = {
                'center': (cx, cy),
                'box': det['box'],
                'first_seen': current_time,
                'last_seen': current_time
            }
            matched_ids.add(next_flame_id)
            target_id = next_flame_id
            next_flame_id += 1

    # 8. Render Bounding Boxes & Update Verification Timers
    for fid, flame_info in list(tracked_flames.items()):
        # Draw box if the flame was seen recently (handles brief dropouts smoothly)
        if current_time - flame_info['last_seen'] <= MAX_DISAPPEARED_SEC:
            x, y, w, h = flame_info['box']
            active_duration = current_time - flame_info['first_seen']

            if active_duration >= ACTIVATION_THRESHOLD_SEC:
                # Verified Fire -> Draw full red bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                label = f"FIRE DETECTED ({active_duration:.1f}s)"
                cv2.putText(frame, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                # Pre-activation -> Yellow box holding the full flame contour
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Verifying... {active_duration:.1f}s", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 9. Clean up expired flame trackers
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
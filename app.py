import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

window_name = 'YOLO Pose & Fire Detection'
cv2.namedWindow(window_name)

REAL_EYE_DISTANCE_M = 0.063
FOCAL_LENGTH_PX = 680
dist_history = []
SMOOTHING_FRAMES = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. Fire Detection (HSV Color Space) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color thresholds for fire (bright red/orange/yellow)
    lower_fire = np.array([0, 120, 180], dtype=np.uint8)
    upper_fire = np.array([35, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # Ignore small light noise
            fire_detected = True
            fx, fy, fw, fh = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE DETECTED!", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if fire_detected:
        cv2.putText(frame, "WARNING: FIRE IN FRAME", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # --- 2. YOLO Person & Pose Tracking ---
    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot(img=frame)

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                box_center_x = int((x1 + x2) / 2)
                box_center_y = int((y1 + y2) / 2)

                distance_m = 0.0
                if len(keypoints) > i:
                    person_kpts = keypoints[i]
                    left_eye = person_kpts[1]
                    right_eye = person_kpts[2]

                    if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                        lx, ly = int(left_eye[0]), int(left_eye[1])
                        rx, ry = int(right_eye[0]), int(right_eye[1])
                        eye_pixel_dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)

                        if eye_pixel_dist > 0:
                            raw_dist = (REAL_EYE_DISTANCE_M * FOCAL_LENGTH_PX) / eye_pixel_dist
                            dist_history.append(raw_dist)
                            if len(dist_history) > SMOOTHING_FRAMES:
                                dist_history.pop(0)
                            distance_m = sum(dist_history) / len(dist_history)

                cv2.circle(annotated_frame, (box_center_x, box_center_y), 6, (0, 0, 255), -1)

                if distance_m > 0:
                    distance_ft = distance_m * 3.28084
                    label = f"Dist: {distance_m:.2f}m ({distance_ft:.1f}ft)"
                else:
                    label = "Dist: Detecting..."

                cv2.putText(annotated_frame, label, (box_center_x - 60, box_center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(window_name, annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
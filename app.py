import cv2
import math
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

window_name = 'YOLO Full Body Tracker'
cv2.namedWindow(window_name)

# --- Calibration Settings (ASUS Zenbook S 14) ---
REAL_EYE_DISTANCE_M = 0.063  # 6.3 cm average distance between human eyes
FOCAL_LENGTH_PX = 680        # Calibrated baseline focal length for 1080p camera

# Moving average smoothing to stop reading jitter
dist_history = []
SMOOTHING_FRAMES = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get pose predictions
    results = model(frame, conf=0.5, verbose=False)

    # 1. Trace the full body skeleton and keypoints automatically
    annotated_frame = results[0].plot()

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])

                # 2. Calculate the center of the bounding box
                box_center_x = int((x1 + x2) / 2)
                box_center_y = int((y1 + y2) / 2)

                # Default fallback distance using eyes
                distance_m = 0.0
                if len(keypoints) > i:
                    person_kpts = keypoints[i]
                    left_eye = person_kpts[1]   # Keypoint 1: Left Eye
                    right_eye = person_kpts[2]  # Keypoint 2: Right Eye

                    # Measure eye distance if visible for stable depth math
                    if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                        lx, ly = int(left_eye[0]), int(left_eye[1])
                        rx, ry = int(right_eye[0]), int(right_eye[1])

                        eye_pixel_dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)

                        if eye_pixel_dist > 0:
                            raw_dist = (REAL_EYE_DISTANCE_M * FOCAL_LENGTH_PX) / eye_pixel_dist
                            
                            # Smooth depth readings
                            dist_history.append(raw_dist)
                            if len(dist_history) > SMOOTHING_FRAMES:
                                dist_history.pop(0)
                            distance_m = sum(dist_history) / len(dist_history)

                # 3. Mark the middle of the bounding box with a red circle
                cv2.circle(annotated_frame, (box_center_x, box_center_y), 6, (0, 0, 255), -1)

                # 4. Display distance text right at the center of the box
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
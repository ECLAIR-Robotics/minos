import cv2
import numpy as np

cap = cv2.VideoCapture(0)
window_name = 'Dedicated Fire & Smoke Tracker'
cv2.namedWindow(window_name)

# Background subtractor to isolate active movement/flicker
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Blur frame to reduce high-frequency camera noise
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    
    # 1. Motion Mask: Fire flickers continuously
    fg_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 2. Color Mask: Fire RGB/HSV characteristics
    # Flame condition: Red > Green > Blue and high luminosity
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(blurred)
    
    # Flame color condition: High Red, Red > Green, Green > Blue, High Value (brightness)
    rgb_fire = (r > 190) & (g > 100) & (r > g) & (g > b)
    hsv_fire = (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 180)
    
    color_mask = np.uint8(rgb_fire & hsv_fire) * 255

    # 3. Combine Motion AND Color (Fire must be moving AND flame-colored)
    fire_mask = cv2.bitwise_and(color_mask, motion_mask)

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

    # Find contours for detected fire regions
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:  # Ignore tiny specks
            fire_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw bold red alert box around fire only
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "FIRE DETECTED", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if fire_detected:
        cv2.putText(frame, "WARNING: ACTIVE FLAME IN FRAME", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
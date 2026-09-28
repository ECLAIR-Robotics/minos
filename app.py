import cv2
import numpy as np

cap = cv2.VideoCapture(0)
window_name = 'Fire Detection'
cv2.namedWindow(window_name)

# Background subtractor to isolate rapid motion & flicker
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=40, detectShadows=False)

# Track motion history to eliminate one-off reflections or quick shifts
motion_history = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Denoise frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # 2. Extract motion (flicker)
    fg_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)

    # 3. Strict Flame Color Conditions (RGB + HSV)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Flames MUST be bright, strong Red component, Red > Green, Green > Blue
    # HSV Hue must strictly fall in the orange/red flame spectrum (0-18)
    rgb_flame = (r > 200) & (g > 120) & (b < 140) & (r > g) & (g > b)
    hsv_flame = (hsv[:, :, 0] <= 18) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 200)
    
    color_mask = np.uint8(rgb_flame & hsv_flame) * 255

    # 4. Require BOTH active flicker AND fire color
    combined_mask = cv2.bitwise_and(color_mask, motion_mask)

    # Morphological filtering to erase small isolated noise points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)

    # 5. Contour Filtering & Bounding Boxes
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Higher area threshold ignores small noise and bright background specks
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check aspect ratio to avoid long horizontal/vertical background stripes
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 4.0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE", (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
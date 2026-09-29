import json
import os
import cv2

# Calibration Settings
KNOWN_DISTANCE_CM = 50.0  # Measured distance from camera to target in cm
KNOWN_WIDTH_CM = 3.0       # Actual width of the flame or target in cm

def update_camera_config(profile_name, focal_length, fire_width_cm):
    """Saves or updates the profile inside camera_config.json."""
    config_file = 'camera_config.json'
    config = {
        "profiles": {},
        "default_fov_deg": 70.0,
        "default_fire_width_cm": 3.0
    }

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading existing JSON: {e}. Overwriting.")

    if "profiles" not in config:
        config["profiles"] = {}

    config["profiles"][profile_name] = {
        "focal_length": round(focal_length, 2),
        "default_fire_width_cm": fire_width_cm
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[SUCCESS] Saved profile '{profile_name}' to '{config_file}'!")

def main():
    profile_name = input("Enter camera profile name (e.g., 'asus_zenbook'): ").strip()
    if not profile_name:
        profile_name = "asus_zenbook"

    print(f"\nPlace target ({KNOWN_WIDTH_CM}cm wide) exactly {KNOWN_DISTANCE_CM}cm from camera.")
    print("When window opens, drag a box around width of target and press ENTER or SPACE.")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'c' to Calibrate, 'q' to Quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Calibration Window', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            r = cv2.selectROI("Calibration Window", frame, False)
            pixel_width = r[2]  # Bounding box width in pixels

            if pixel_width > 0:
                focal_length = (pixel_width * KNOWN_DISTANCE_CM) / KNOWN_WIDTH_CM
                print(f"\nMeasured Width: {pixel_width} px")
                print(f"Calculated Focal Length: {focal_length:.2f}")

                update_camera_config(profile_name, focal_length, KNOWN_WIDTH_CM)
                break
            else:
                print("[ERROR] Invalid selection width. Try again.")

        elif key == ord('q'):
            print("Calibration canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
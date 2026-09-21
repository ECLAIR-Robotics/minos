import cv2
import math
import json
import os
from collections import deque
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = 'yolov8n-pose.pt'
CALIBRATION_FILE = 'calibration.json'
SMOOTHING_WINDOW = 7          # frames used for median outlier rejection
EMA_ALPHA = 0.35              # exponential moving average weight (higher = more responsive)
CONF_THRESHOLD = 0.5          # minimum keypoint confidence to trust a keypoint
YAW_REJECT_RATIO = 1.6        # if left/right sub-distances differ by more than this ratio, downweight cue

# COCO keypoint indices (YOLOv8-pose)
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# ============================================================
# CALIBRATION STORAGE
# ============================================================
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_calibration(cal):
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"[calibration] saved to {CALIBRATION_FILE}: {cal}")


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def kp_ok(kp):
    """kp = [x, y, conf]"""
    return kp is not None and kp[2] > CONF_THRESHOLD

def pixel_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def frontal_weight(nose, left, right):
    """
    Heuristic: compare nose-to-left vs nose-to-right pixel distance.
    A big asymmetry means the person is turned (yaw), which foreshortens
    the measurement -> lower confidence in this cue.
    Returns a weight in [0.1, 1.0].
    """
    if not kp_ok(nose):
        return 0.6  # can't check yaw, assume moderate trust
    dl = pixel_dist(nose, left)
    dr = pixel_dist(nose, right)
    if min(dl, dr) < 1e-3:
        return 0.1
    ratio = max(dl, dr) / min(dl, dr)
    if ratio >= YAW_REJECT_RATIO:
        return 0.1
    # linearly scale weight down as ratio grows from 1.0 -> YAW_REJECT_RATIO
    return max(0.1, 1.0 - (ratio - 1.0) / (YAW_REJECT_RATIO - 1.0))


# ============================================================
# PER-CUE DISTANCE ESTIMATORS
# distance = K / pixel_measurement
# K is solved during calibration as: K = known_real_distance * pixel_measurement_at_that_distance
# This bakes in both focal length AND this person's actual body dimensions
# in one step, no separate measuring of eye/shoulder width required.
# ============================================================
def estimate_cues(person_kpts, cal):
    """Returns list of (distance_m, weight, name) for available cues."""
    cues = []
    nose = person_kpts[NOSE]
    l_eye, r_eye = person_kpts[L_EYE], person_kpts[R_EYE]
    l_sh, r_sh = person_kpts[L_SHOULDER], person_kpts[R_SHOULDER]
    l_hip, r_hip = person_kpts[L_HIP], person_kpts[R_HIP]

    # --- Eyes ---
    if kp_ok(l_eye) and kp_ok(r_eye) and 'eyes' in cal:
        d = pixel_dist(l_eye, r_eye)
        if d > 1e-3:
            dist = cal['eyes'] / d
            w = frontal_weight(nose, l_eye, r_eye) * 1.0  # eyes are the most reliable cue when frontal
            cues.append((dist, w, 'eyes'))

    # --- Shoulders ---
    if kp_ok(l_sh) and kp_ok(r_sh) and 'shoulders' in cal:
        d = pixel_dist(l_sh, r_sh)
        if d > 1e-3:
            dist = cal['shoulders'] / d
            w = frontal_weight(nose, l_sh, r_sh) * 0.8  # slightly less reliable (clothing, posture)
            cues.append((dist, w, 'shoulders'))

    # --- Hips (fallback, e.g. when face isn't visible) ---
    if kp_ok(l_hip) and kp_ok(r_hip) and 'hips' in cal:
        d = pixel_dist(l_hip, r_hip)
        if d > 1e-3:
            dist = cal['hips'] / d
            w = frontal_weight(nose, l_hip, r_hip) * 0.6  # least reliable (occlusion, loose clothing)
            cues.append((dist, w, 'hips'))

    return cues

def fuse_cues(cues):
    if not cues:
        return None, []
    # reject cues that disagree wildly with the group median (bad keypoint pass-through)
    dists = sorted(d for d, w, n in cues)
    median = dists[len(dists) // 2]
    filtered = [(d, w, n) for d, w, n in cues if abs(d - median) / median < 0.35] or cues

    total_w = sum(w for _, w, _ in filtered)
    if total_w <= 0:
        return None, []
    fused = sum(d * w for d, w, _ in filtered) / total_w
    return fused, filtered


# ============================================================
# TEMPORAL SMOOTHING: median outlier rejection -> EMA
# ============================================================
class DistanceSmoother:
    def __init__(self):
        self.raw_history = deque(maxlen=SMOOTHING_WINDOW)
        self.ema = None

    def update(self, raw_value):
        self.raw_history.append(raw_value)
        sorted_vals = sorted(self.raw_history)
        median_val = sorted_vals[len(sorted_vals) // 2]
        if self.ema is None:
            self.ema = median_val
        else:
            self.ema = EMA_ALPHA * median_val + (1 - EMA_ALPHA) * self.ema
        return self.ema


# ============================================================
# CALIBRATION ROUTINE
# ============================================================
def run_calibration(person_kpts):
    """
    Prompts the user for the real distance they're standing at (measure it
    with a tape measure first) and solves K for each visible cue.
    Stand facing the camera straight-on with shoulders squared for best results.
    Distance is entered and stored in FEET throughout.
    """
    try:
        known_distance = float(input("\n[calibration] Enter your measured distance to camera in feet (e.g. 5): "))
    except ValueError:
        print("[calibration] Invalid number, aborting calibration.")
        return {}

    cal = {}
    nose = person_kpts[NOSE]
    l_eye, r_eye = person_kpts[L_EYE], person_kpts[R_EYE]
    l_sh, r_sh = person_kpts[L_SHOULDER], person_kpts[R_SHOULDER]
    l_hip, r_hip = person_kpts[L_HIP], person_kpts[R_HIP]

    if kp_ok(l_eye) and kp_ok(r_eye):
        d = pixel_dist(l_eye, r_eye)
        cal['eyes'] = known_distance * d
        print(f"[calibration] eyes: pixel_dist={d:.1f} -> K={cal['eyes']:.2f}")
    if kp_ok(l_sh) and kp_ok(r_sh):
        d = pixel_dist(l_sh, r_sh)
        cal['shoulders'] = known_distance * d
        print(f"[calibration] shoulders: pixel_dist={d:.1f} -> K={cal['shoulders']:.2f}")
    if kp_ok(l_hip) and kp_ok(r_hip):
        d = pixel_dist(l_hip, r_hip)
        cal['hips'] = known_distance * d
        print(f"[calibration] hips: pixel_dist={d:.1f} -> K={cal['hips']:.2f}")

    if not cal:
        print("[calibration] No usable keypoints visible — face the camera and try again.")
    return cal


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    window_name = 'YOLO Full Body Tracker'
    cv2.namedWindow(window_name)

    calibration = load_calibration()
    smoothers = {}  # per-detection-index smoother (simple; assumes ~stable ordering for single-person use)

    calibrate_requested = False
    calibration_in_progress = False  # debounce: blocks re-trigger from queued key repeats

    print("Press 'c' to calibrate (stand at a known, measured distance facing the camera).")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        status_text = "CALIBRATED" if calibration else "NOT CALIBRATED - press 'c'"
        cv2.putText(annotated_frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if calibration else (0, 0, 255), 2)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                box_center_x = int((x1 + x2) / 2)
                box_center_y = int((y1 + y2) / 2)

                if len(keypoints) <= i:
                    continue
                person_kpts = keypoints[i]

                if calibrate_requested and not calibration_in_progress and i == 0:
                    calibration_in_progress = True
                    new_cal = run_calibration(person_kpts)
                    if new_cal:
                        calibration = new_cal
                        save_calibration(calibration)
                    calibrate_requested = False

                    # Flush any keypresses (incl. repeats of 'c') queued while
                    # input() was blocking, so they don't retrigger calibration.
                    flush_count = 0
                    while cv2.waitKey(1) != -1 and flush_count < 50:
                        flush_count += 1
                    calibration_in_progress = False

                cv2.circle(annotated_frame, (box_center_x, box_center_y), 6, (0, 0, 255), -1)

                if calibration:
                    cues = estimate_cues(person_kpts, calibration)
                    fused_dist, used_cues = fuse_cues(cues)

                    if fused_dist is not None:
                        smoother = smoothers.setdefault(i, DistanceSmoother())
                        smooth_dist_ft = smoother.update(fused_dist)  # fused_dist is in feet, since K was calibrated in feet

                        cue_names = "+".join(n for _, _, n in used_cues)
                        label = f"{smooth_dist_ft:.1f}ft [{cue_names}]"
                    else:
                        label = "Dist: no usable cues"
                else:
                    label = "Dist: press 'c' to calibrate"

                cv2.putText(annotated_frame, label, (box_center_x - 90, box_center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('c'):
            calibrate_requested = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

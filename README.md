# 🔥 Real-Time Fire Detection & Distance Tracker

A lightweight OpenCV computer vision system that detects fire in real time, reduces false positives using multi-spectral color and temporal flicker analysis, tracks flames, identifies the flame root/base, and estimates physical distance using monocular camera geometry.

---

## ⚡ Features

* **Strict Color & Motion Filtering**

  * Uses RGB color dominance and HSV thresholding to distinguish flames from ambient lamps and other yellow/static objects.

* **Temporal Flicker Analysis**

  * Analyzes pixel intensity variation over rolling frame buffers.
  * Uses standard deviation thresholds of approximately `18.0 ≤ std ≤ 100.0` to identify flame-like temporal behavior.

* **Confidence Scoring**

  * Provides a real-time `0–100%` detection confidence score.
  * Uses approximately `0.5s` of activation verification to reduce false positives.

* **Flame Base Detection**

  * Identifies the bottom/root region of the detected flame.
  * Displays the detected flame base using a cyan marker.

* **Monocular Distance Estimation**

  * Estimates the physical distance to the detected flame in `cm` / `m`.
  * Supports multiple camera profiles through `camera_config.json`.
  * Includes an automatic `70°` horizontal FOV fallback when no calibration profile is available.

---

## 🚀 Quick Start

The application can be run immediately without manually calibrating the camera.

If no camera profile is available, the system automatically estimates the camera's focal length using a standard `70°` horizontal Field of View.

### Run the Application

```powershell
venv\Scripts\python.exe app.py
```

---

# 🎯 Camera Calibration

Camera calibration only needs to be performed **once per camera**.

Calibration allows the application to use a measured focal length rather than the default `70°` FOV estimate, improving distance measurements.

## 1. Setup

You will need:

* A ruler or measuring tape
* A target object that is exactly **3 cm wide**
* A webcam

Place the target:

* **Width:** `3 cm`
* **Distance from webcam:** `50 cm`

Make sure the object is positioned perpendicular to the camera as accurately as possible.

---

## 2. Run the Calibration Script

From the project directory, run:

```powershell
venv\Scripts\python.exe calibrate.py
```

A webcam window will appear.

---

## 3. Complete the Calibration

### Step 1 — Name Your Profile

The terminal will ask for a profile name.

For example:

```text
asus_zenbook
```

Press **Enter**.

---

### Step 2 — Freeze the Frame

Look at the webcam window and press:

```text
c
```

This freezes the current frame and activates the mouse selection tool.

---

### Step 3 — Select the Target

Using your mouse:

1. Click and hold on one side of the `3 cm` target.
2. Drag across the target.
3. Release the mouse when the bounding box covers the target's width.

Try to make the box as tight and accurate as possible.

---

### Step 4 — Save the Calibration

Press:

```text
ENTER
```

or

```text
SPACEBAR
```

The calibration script will calculate the camera's focal length and save the resulting profile to:

```text
camera_config.json
```

---

### Step 5 — Exit Calibration

Press:

```text
q
```

to close the calibration window.

---

# ⚙️ Enable Your Camera Profile

After calibration, open `app.py` and find:

```python
SELECTED_PROFILE = "asus_zenbook"
```

Replace `"asus_zenbook"` with the profile name you created during calibration.

For example:

```python
SELECTED_PROFILE = "asus_zenbook"
```

Then run the detection application:

```powershell
venv\Scripts\python.exe app.py
```

The application will now use the calibrated camera profile for distance estimation.

---

# ⌨️ Calibration Controls

| Key / Input          | Action                                                       |
| -------------------- | ------------------------------------------------------------ |
| `c`                  | Freeze the current frame and activate the mouse box selector |
| Mouse Click + Drag   | Draw a bounding box around the target                        |
| `ENTER` / `SPACEBAR` | Confirm the selection and save the calibration               |
| `c` again            | Reset the box selection if you made a mistake                |
| `q`                  | Exit the calibration tool                                    |

---

# 📁 Project Structure

A typical project layout looks like:

```text
project/
│
├── app.py
├── calibrate.py
├── camera_config.json
│
├── venv/
│
└── README.md
```

---

# 📏 Distance Estimation

The system uses monocular camera geometry to estimate the distance between the camera and the detected flame.

The calibration process determines the camera's effective focal length using a known object width and known distance.

Once calibrated, the system can use the apparent size of the detected flame to estimate its distance from the camera.

If no calibrated profile is selected, the application falls back to an estimated:

```text
70° horizontal FOV
```

For the most accurate distance measurements, calibration is recommended.

---

# 🧠 Detection Pipeline

The fire detection system combines multiple signals rather than relying on color alone:

```text
Camera Input
     │
     ▼
Color Filtering
(RGB + HSV)
     │
     ▼
Motion / Candidate Detection
     │
     ▼
Temporal Flicker Analysis
     │
     ▼
Confidence Scoring
     │
     ▼
Flame Tracking
     │
     ├──► Flame Base Detection
     │
     └──► Distance Estimation
```

This multi-stage approach helps distinguish actual flames from static objects that have similar colors.

---

# 📝 Notes

* Distance estimation depends on accurate camera calibration.
* The `3 cm` calibration target should be measured carefully.
* The target should be exactly `50 cm` from the camera during calibration.
* Different cameras should use separate profiles in `camera_config.json`.
* The automatic `70°` FOV mode is intended as a fallback and may be less accurate than calibration.
* Detection confidence is an estimate based on the system's color, motion, and temporal analysis rather than a guaranteed probability of fire.

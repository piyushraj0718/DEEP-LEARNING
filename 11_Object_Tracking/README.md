# 11 — Object Tracking

This module covers classical and deep learning-based object tracking algorithms using OpenCV and YOLO11. It is a **standalone module** with its own virtual environment because it requires a specific version of `opencv-contrib-python` that includes the legacy tracking APIs.

---

## Why a Separate Virtual Environment?

The traditional trackers in this module depend on `cv2.legacy.*` APIs available only in `opencv-contrib-python`. This version conflicts with the standard `opencv-python` used across the rest of the repository. The isolated `venv/` keeps these dependencies from interfering with the other modules.

---

## Setup

```bash
# From the 11_Object_Tracking/ directory

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux / macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Module Structure

```text
11_Object_Tracking/
│
├── requirements.txt              # Module-specific dependencies
├── README.md                     # This file
│
├── Traditional_Trackers/         # Classical OpenCV legacy trackers
│   ├── 01_Boosting_Tracker.py
│   ├── 02_CSRT_Tracker.py
│   ├── 03_KCF_Tracker.py
│   ├── 04_MedianFlow_Tracker.py
│   ├── 05_MIL_Tracker.py
│   ├── 06_MOSSE_Tracker.py
│   └── 07_TLD_Tracker.py
│
├── GOTURN/                       # Deep learning tracker (Caffe-based)
│   ├── goturn.py
│   ├── goturn.caffemodel         # ~400 MB pre-trained weights
│   └── goturn.prototxt           # Network architecture definition
│
├── YOLO_Tracking/                # YOLO11 + ByteTrack multi-object tracking
│   ├── stream.py                 # Real-time webcam tracking
│   ├── object_tracker.py         # Video file tracking with save output
│   ├── yolo11n.pt                # YOLO11 nano model weights
│   ├── pwvid_NQdU7EJ2.mp4        # Sample video 1
│   └── ult_lrNqIsgc.mp4          # Sample video 2
│
└── venv/                         # Module-specific virtual environment (not tracked in Git)
```

---

## Tracker Reference

### Traditional Trackers (`opencv-contrib-python` required)

| # | Tracker | Speed | Accuracy | Handles Occlusion | Best For |
|---|---------|-------|----------|-------------------|----------|
| 01 | **Boosting** | Medium | Low | ✗ | Educational baseline |
| 02 | **CSRT** | Slow | High | Partial | Precision tracking |
| 03 | **KCF** | Fast | Medium | ✗ | Real-time, moderate motion |
| 04 | **MedianFlow** | Fast | Medium | ✗ | Slow, predictable motion |
| 05 | **MIL** | Medium | Medium | Partial | Loose ROI initialization |
| 06 | **MOSSE** | Very Fast | Low–Medium | ✗ | Embedded/real-time systems |
| 07 | **TLD** | Slow | Medium | ✓ | Long-term tracking |

### Deep Learning Trackers

| Tracker | Framework | Model | Best For |
|---------|-----------|-------|----------|
| **GOTURN** | OpenCV + Caffe | AlexNet-based regression | Novel object categories, offline |
| **YOLO11 + ByteTrack** | Ultralytics | YOLOv11n | Real-time multi-object tracking |

---

## Running a Tracker

All traditional trackers follow the same pattern:

1. Run the script
2. A window opens showing the first webcam frame
3. Draw a bounding box around the object to track and press **Enter**
4. The tracker begins. Press **ESC** to exit.

```bash
# Example: run KCF tracker
python Traditional_Trackers/03_KCF_Tracker.py

# Run GOTURN tracker (requires model files in GOTURN/ folder)
python GOTURN/goturn.py

# Run YOLO real-time webcam tracking
python YOLO_Tracking/stream.py

# Run YOLO video file tracking
python YOLO_Tracking/object_tracker.py
```

---

## GOTURN Model Files

GOTURN requires two large model files (~400 MB total) that are not included in the repository. Download them and place them in the `GOTURN/` folder:

- `goturn.prototxt` — network architecture
- `goturn.caffemodel` — pre-trained weights

These are available from the [OpenCV Extra repository](https://github.com/opencv/opencv_extra).

---

## Dependencies

See `requirements.txt` for the full list. Key packages:

```
opencv-contrib-python   # Includes legacy trackers and GOTURN
ultralytics             # YOLO11 + ByteTrack
numpy
```

---

## Key Concepts

- **ROI Selection**: All traditional trackers require manual bounding box selection on the first frame using `cv2.selectROI()`
- **Tracking vs Detection**: Trackers follow an object; they do not detect new objects. YOLO combines both.
- **ByteTrack**: The tracking algorithm used by YOLO scripts — assigns persistent IDs across frames using IoU-based matching
- **Legacy Module**: OpenCV's `cv2.legacy.*` namespace holds the classical tracker implementations

---

## Related Modules

- **07 — Object Detection**: R-CNN and Faster R-CNN for detection-based approaches
- **06 — Classic CNN Architectures**: YOLO-based image classification (foundation for tracking)

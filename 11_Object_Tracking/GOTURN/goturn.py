"""
GOTURN Object Tracker (Generic Object Tracking Using Regression Networks)

This implementation uses the GOTURN tracking algorithm from OpenCV's tracking
module. GOTURN is a deep learning-based tracker that uses a pre-trained
Caffe neural network (AlexNet-based) to regress the bounding box of the
tracked object in each frame.

How it works:
    Unlike traditional trackers that learn from a single video sequence,
    GOTURN is trained offline on a large dataset of videos. At runtime, it
    takes two inputs:
      1. The previous frame cropped around the object (to encode appearance)
      2. The current frame cropped around the search region (to locate object)
    The neural network directly regresses the four bounding box coordinates
    in the current frame, making it extremely fast at inference time since
    no online learning occurs.

Advantages:
    - Fast at runtime since the network is pre-trained (no online updates)
    - Generalizes to novel object categories without retraining
    - Robust to appearance variation since it was trained on diverse videos
    - Deep learning-powered localization accuracy

Limitations:
    - Requires large Caffe model files (~370 MB)
    - Less accurate than modern deep trackers (SiamRPN, OSTrack, etc.)
    - Cannot recover after losing the object (no re-detection)
    - Fixed model: does not adapt to the specific tracked object online
    - Performance limited by the quality of the pre-trained network

Typical Use Cases:
    - Tracking novel object categories without any fine-tuning
    - Systems where inference speed is more important than accuracy
    - Introduction to deep learning-based tracking methods

Model Files Required:
    - goturn.prototxt     (network architecture definition)
    - goturn.caffemodel   (pre-trained weights, ~370 MB)
    Both files must be in the GOTURN/ folder alongside this script.

Requirements:
    - OpenCV with tracking module: opencv-contrib-python
    - Caffe model files (see above)
"""

import cv2
import os
import sys
import time

# Define GOTURN model paths relative to this file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT   = os.path.join(_DIR, "goturn.prototxt")
CAFFEMODEL = os.path.join(_DIR, "goturn.caffemodel")

def validate_goturn_model():
    if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
        print("\n❌ GOTURN model files not found.")
        print("Expected files:")
        print(f"  - {PROTOTXT}")
        print(f"  - {CAFFEMODEL}")
        print("Please place them in the GOTURN/ folder alongside this script.\n")
        sys.exit(1)

def main():
    validate_goturn_model()

    # Set up GOTURN parameters
    params = cv2.TrackerGOTURN_Params()
    params.modelTxt = PROTOTXT
    params.modelBin = CAFFEMODEL

    # Create GOTURN tracker
    tracker = cv2.TrackerGOTURN_create(params)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        return

    # Let user select ROI
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Initialize tracker
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track object
        start = time.time()
        success, bbox = tracker.update(frame)
        end = time.time()

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"GOTURN: {1000 * (end - start):.1f} ms"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Lost", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("GOTURN Tracker", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

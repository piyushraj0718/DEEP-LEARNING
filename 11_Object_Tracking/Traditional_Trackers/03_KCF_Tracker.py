"""
KCF Object Tracker (Kernelized Correlation Filters)

This implementation uses the KCF tracking algorithm from OpenCV's legacy
tracking module. KCF is a fast and efficient correlation-filter-based tracker
that strikes a strong balance between speed and accuracy.

How it works:
    KCF exploits the properties of circulant matrices to train a ridge
    regression classifier in the Fourier domain. By generating many virtual
    training samples through cyclic shifts of the base patch, it learns a
    discriminative filter extremely efficiently. The kernel trick (typically
    a Gaussian kernel) allows it to operate in a high-dimensional feature
    space without explicit computation.

Advantages:
    - Significantly faster than CSRT and Boosting trackers
    - Good accuracy for objects moving at moderate speeds
    - Low memory footprint
    - Well-studied and widely benchmarked

Limitations:
    - Does not handle scale changes natively (fixed bounding box size)
    - Struggles with fast non-linear motion or abrupt direction changes
    - Cannot recover after the object leaves the frame
    - Less robust to occlusion than CSRT

Typical Use Cases:
    - Real-time tracking on CPU-constrained hardware
    - Tracking objects with consistent size and moderate motion
    - Robotics and embedded vision systems where speed is critical

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

# Define the tracker (you can try 'csrt', 'kcf', 'mil', 'tld', 'medianflow', 'mosse')
tracker_type = 'kcf'
tracker = cv2.legacy.TrackerKCF_create()

# Load video from file or webcam
video = cv2.VideoCapture(0)  # Use 0 for webcam or replace with "video.mp4"

# Read the first frame
ret, frame = video.read()
if not ret:
    print("Failed to read video")
    exit()

# Select the bounding box (ROI) for the object to track
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Initialize the tracker with the first frame and selected bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    # Draw bounding box
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Object Tracking", frame)

    # Exit with ESC
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()

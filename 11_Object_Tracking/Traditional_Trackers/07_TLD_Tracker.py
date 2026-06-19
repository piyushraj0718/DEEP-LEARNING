"""
TLD Object Tracker (Tracking-Learning-Detection)

This implementation uses the TLD tracking algorithm from OpenCV's legacy
tracking module. TLD is unique in that it combines three independent
components — a tracker, a detector, and a learning module — into a unified
framework, enabling long-term tracking with re-detection capability.

How it works:
    TLD decomposes the tracking problem into three tasks:
      1. Tracker: Uses optical flow to estimate the object's position
         frame-to-frame.
      2. Detector: Scans the entire frame using a learned appearance model
         (cascade of classifiers) to find the object independently of the
         tracker.
      3. Learning (P-N Learning): Automatically generates positive and
         negative training examples by analyzing tracker and detector
         failures, then updates the detector to improve future performance.
    When the tracker fails, the detector can re-acquire the object; when the
    detector provides a better estimate, it corrects the tracker.

Advantages:
    - Can re-detect the object after complete loss or occlusion
    - Suitable for long-term tracking across many frames
    - Continually improves its appearance model during tracking
    - Handles objects that disappear and reappear in the scene

Limitations:
    - Computationally expensive compared to KCF and MOSSE
    - Can generate false positives in cluttered scenes
    - Unstable initialization can degrade performance significantly
    - Slow on high-resolution video without hardware acceleration

Typical Use Cases:
    - Long-duration surveillance tracking
    - Scenarios where objects may leave and re-enter the frame
    - Tracking across scene cuts or temporary full occlusions
    - Applications requiring continuous re-detection capability

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

# Define the tracker
tracker_type = 'tld'
tracker = cv2.legacy.TrackerTLD_create()

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

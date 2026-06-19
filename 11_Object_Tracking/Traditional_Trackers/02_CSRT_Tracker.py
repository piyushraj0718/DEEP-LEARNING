"""
CSRT Object Tracker (Discriminative Correlation Filter with Channel and Spatial Reliability)

This implementation uses the CSRT tracking algorithm from OpenCV's legacy
tracking module. CSRT is one of the most accurate traditional trackers and
is well-suited for real-world tracking scenarios.

How it works:
    CSRT extends the standard Discriminative Correlation Filter (DCF) by
    introducing a channel and spatial reliability map. It selects only the
    most reliable spatial regions for filter learning, which helps focus
    the tracker on the most discriminative parts of the object. Multiple
    feature channels (HOG, Color Names) are fused to improve robustness.

Advantages:
    - High accuracy, especially for objects that change scale or orientation
    - Robust to partial occlusion and background clutter
    - Handles irregular object shapes better than most trackers
    - Good at recovering from brief occlusions

Limitations:
    - Slower than KCF and MOSSE due to its higher computational cost
    - Not suitable for real-time tracking on low-end hardware without GPU
    - Can struggle with very fast motion or full occlusion

Typical Use Cases:
    - Precision tracking in surveillance and security applications
    - Tracking objects that undergo scale changes or partial occlusion
    - Any scenario where accuracy is prioritized over speed

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

# Define the tracker
tracker_type = 'csrt'
tracker = cv2.legacy.TrackerCSRT_create()

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

"""
MedianFlow Object Tracker

This implementation uses the MedianFlow tracking algorithm from OpenCV's
legacy tracking module. MedianFlow is notable for its ability to detect
tracking failures, making it more reliable in scenarios where the object
temporarily disappears or the tracker loses confidence.

How it works:
    MedianFlow tracks the object by predicting a dense set of points on the
    object using Lucas-Kanade optical flow in both the forward (t → t+1) and
    backward (t+1 → t) directions. The forward-backward error — the distance
    between the original points and the back-tracked points — is computed for
    each point. The median of these errors is used to estimate the reliability
    of the tracking. Points with high error are discarded, and the remaining
    points determine the new bounding box.

Advantages:
    - Can detect and report tracking failures (unlike most other trackers)
    - Works well for slow, predictable motion
    - Fast and computationally lightweight
    - Provides a confidence metric for the tracking result

Limitations:
    - Fails on fast-moving objects where optical flow becomes unreliable
    - Does not handle occlusion well
    - Performance degrades significantly with motion blur
    - Not suitable for highly textured or uniform-colored objects

Typical Use Cases:
    - Tracking slowly moving objects where failure detection is important
    - Scenarios requiring a tracker that can gracefully report "lost"
    - Aerial or drone footage with steady camera motion

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

# Define the tracker
tracker_type = 'medianflow'
tracker = cv2.legacy.TrackerMedianFlow_create()

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

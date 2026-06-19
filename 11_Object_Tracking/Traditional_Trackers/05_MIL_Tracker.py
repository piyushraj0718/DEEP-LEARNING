"""
MIL Object Tracker (Multiple Instance Learning)

This implementation uses the MIL tracking algorithm from OpenCV's legacy
tracking module. MIL applies a Multiple Instance Learning framework to
tracking, which makes it more robust to imprecise bounding box initialization
compared to traditional supervised trackers.

How it works:
    Instead of treating the object's bounding box as a single positive
    training example, MIL considers a "bag" of image patches centered around
    the tracked position. The bag is labeled positive if at least one patch
    in it contains the object. This relaxed labeling allows the tracker to
    handle slight localization errors during training. An online classifier
    is updated each frame using these bags to predict the object's next
    position.

Advantages:
    - More robust to initialization errors than single-instance trackers
    - Handles moderate changes in object appearance over time
    - Less sensitive to exact bounding box boundaries
    - Performs better than Boosting in cluttered backgrounds

Limitations:
    - Slower than KCF and MOSSE
    - Can accumulate errors over long tracking sequences (drift)
    - Does not explicitly handle occlusion or scale changes
    - May lose the object in scenes with many similar-looking distractors

Typical Use Cases:
    - Tracking with loosely defined initial regions of interest
    - Scenarios where the initial bounding box may not be perfectly aligned
    - Research and comparison benchmarks for classical tracking

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

tracker_type = 'mil'
tracker = cv2.legacy.TrackerMIL_create()

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

"""
Boosting Object Tracker

This implementation uses the Boosting tracking algorithm from OpenCV's legacy
tracking module. It is based on the AdaBoost algorithm and uses an online
learning approach to track a manually selected object across video frames.

How it works:
    The tracker trains a classifier on the appearance of the selected object
    (positive samples) and the background region around it (negative samples).
    In each frame, the classifier scans a search window around the last known
    position and selects the region with the highest score as the new location.
    The classifier is continuously updated as tracking proceeds.

Advantages:
    - Simple and well-understood algorithm
    - Adaptive: updates its model during tracking
    - Reasonable performance on objects with consistent appearance

Limitations:
    - Lower accuracy compared to modern trackers (KCF, CSRT)
    - Prone to drift when the object undergoes rapid motion or partial occlusion
    - Cannot recover after complete loss of the object
    - Slow compared to correlation-filter-based trackers

Typical Use Cases:
    - Educational demonstrations of classical tracking
    - Slow-moving objects in controlled environments
    - Baseline comparison against more advanced trackers

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

def main():
    # Initialize Boosting Tracker
    tracker = cv2.legacy.TrackerBoosting_create()

    # Open webcam or video file
    video = cv2.VideoCapture(0)  # Replace 0 with path to video file if needed

    # Read the first frame
    ret, frame = video.read()
    if not ret:
        print("Failed to read from video source")
        video.release()
        return

    # Select ROI (object to track)
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Initialize the tracker with the selected ROI
    tracker.init(frame, bbox)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Update the tracker
        success, bbox = tracker.update(frame)

        if success:
            # Tracking success: draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display result
        cv2.imshow("Boosting Tracker", frame)

        # Exit on ESC key
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

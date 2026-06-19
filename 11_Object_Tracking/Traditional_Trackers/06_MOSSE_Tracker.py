"""
MOSSE Object Tracker (Minimum Output Sum of Squared Error)

This implementation uses the MOSSE tracking algorithm from OpenCV's legacy
tracking module. MOSSE is one of the fastest classical trackers available
and is particularly well-suited for real-time applications where low latency
is critical.

How it works:
    MOSSE learns a correlation filter that, when convolved with the input
    image patch, produces a desired Gaussian-shaped response peaked at the
    object's center. The filter is initialized in the first frame and updated
    incrementally in subsequent frames using a running average of the
    numerator and denominator in the frequency domain. This formulation
    allows very fast filter updates with minimal computation.

Advantages:
    - Extremely fast — can run at hundreds of frames per second
    - Very low computational cost, suitable for embedded systems
    - Stable tracking on objects with consistent grayscale texture
    - Simple and efficient frequency-domain implementation

Limitations:
    - Operates on grayscale only; does not use color information
    - Poor performance on objects with little texture or uniform appearance
    - Vulnerable to occlusion and rapid illumination changes
    - Fixed scale: cannot adapt to changes in object size

Typical Use Cases:
    - High-speed real-time tracking applications
    - Robotics and autonomous systems with limited processing power
    - Tracking fast-moving objects where latency matters most
    - Baseline for evaluating more complex correlation-filter trackers

Requirements:
    - OpenCV with legacy module: opencv-contrib-python
"""

import cv2

# Define the tracker (you can try 'csrt', 'kcf', 'mil', 'tld', 'medianflow', 'mosse')
tracker_type = 'mosse'
tracker = cv2.legacy.TrackerMOSSE_create()

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

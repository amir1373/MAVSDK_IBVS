#!/usr/bin/env python3
import cv2
import numpy as np
from pupil_apriltags import Detector
import math

STREAM_URL = (
    "udp://0.0.0.0:2032"
    "?overrun_nonfatal=1"
    "&fifo_size=500000"
    "&flags=low_delay"
)

detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0
)

cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open FPV stream")
    exit()

print("▶ Hold an AprilTag in front of the camera...")
print("▶ Press ESC to exit.\n")

def classify_angle(angle):
    """Return nearest orientation in degrees."""
    angle = angle % 360
    if abs(angle - 0) < 25:   return "Correct (0°)"
    if abs(angle - 90) < 25:  return "Rotated 90° Right"
    if abs(angle - 180) < 25: return "Upside-down (180°)"
    if abs(angle - 270) < 25: return "Rotated 90° Left"
    return "Unknown angle"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    if results:
        tag = results[0]

        # Draw corners
        for i in range(4):
            p1 = tuple(tag.corners[i].astype(int))
            p2 = tuple(tag.corners[(i+1)%4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Compute orientation angle of tag in image
        # Using the vector from corner 0 → corner 1
        c0 = tag.corners[0]
        c1 = tag.corners[1]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360

        orientation = classify_angle(angle)

        cv2.putText(frame, 
                    f"Angle: {angle:.1f} deg", 
                    (10,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0,255,255), 2)

        cv2.putText(frame, 
                    f"Camera orientation: {orientation}", 
                    (10,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0,128,255), 2)

    cv2.imshow("Camera Orientation Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

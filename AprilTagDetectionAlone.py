#!/usr/bin/env python3
import cv2
import numpy as np
from pupil_apriltags import Detector

# AprilTag detector
detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect tags
    results = detector.detect(gray)

    h, w = frame.shape[:2]
    cx = w // 2
    cy = h // 2

    # draw screen center
    cv2.circle(frame, (cx, cy), 6, (0,255,255), -1)

    for r in results:
        # Tag center
        (tcx, tcy) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (tcx, tcy), 6, (0,0,255), -1)

        # Tag corners
        corners = r.corners.astype(int)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i+1)%4])
            cv2.line(frame, p1, p2, (0,255,0), 2)

        # Draw line from screen center → tag
        cv2.line(frame, (cx,cy), (tcx,tcy), (255,0,0), 2)

        # pixel offset
        dx = tcx - cx
        dy = tcy - cy
        cv2.putText(frame, f"dx={dx}, dy={dy}",
                    (tcx+10, tcy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)

    cv2.imshow("AprilTag Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

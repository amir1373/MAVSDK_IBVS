#!/usr/bin/env python3
import os
import threading
import time

import cv2


# Match this URL everywhere else in the project.
# Keep the UDP FIFO small; big FIFOs make 1080p feel delayed because old frames queue up.
stream_url = (
    "udp://0.0.0.0:2031"
    "?overrun_nonfatal=1"
    "&fifo_size=50000"
    "&flags=low_delay"
)


class LatestFrameCamera:
    def __init__(self, url):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;0",
        )
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.ret = False
        self.frames_received = 0
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def is_opened(self):
        return self.cap.isOpened()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = True
                self.frame = frame
                self.frames_received += 1

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.frames_received
            return self.ret, self.frame.copy(), self.frames_received

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def main():
    cam = LatestFrameCamera(stream_url)
    if not cam.is_opened():
        print("Unable to open stream")
        cam.release()
        return

    print("Low-latency stream started. Press 'q' or ESC to quit.")
    print(f"URL: {stream_url}")

    displayed = 0
    t0 = time.time()

    while True:
        ret, frame, received = cam.read_latest()
        if not ret:
            time.sleep(0.002)
            continue

        displayed += 1
        elapsed = max(time.time() - t0, 1e-6)
        display_fps = displayed / elapsed

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"{w}x{h} display:{display_fps:.1f}fps received:{received}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Low-Latency UDP Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

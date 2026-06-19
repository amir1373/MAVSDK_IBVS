#!/usr/bin/env python3
"""
MAVSDK + Fast FPV + AprilTag + SAFE PID IBVS
--------------------------------------------
- PX4 NED body frame (X forward, Y right, Z down)
- OFFBOARD-only takeoff (velocity-based climb)
- AprilTag-based PID IBVS (slow & controllable)
- Low-latency FPV via threaded FastCamera
- Camera feed is flipped 180° (frame = cv2.flip(frame, -1))
- Stops approaching around ~1.5 m (via tag area threshold)
- Hard distance safety cap
- If IBVS is ON and tag is lost -> hover (vx=vy=vz=0)
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
from pupil_apriltags import Detector


# ============================================================
# PID CONTROLLER
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, windup=2000.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.windup = windup

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error, dt):
        P = self.kp * error

        self.integral += error * dt
        self.integral = max(min(self.integral, self.windup), -self.windup)
        I = self.ki * self.integral

        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        D = self.kd * derivative

        self.previous_error = error
        return P + I + D


# ============================================================
# FAST CAMERA (THREAD)
# ============================================================
class FastCamera:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.ret = True
            self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except:
            pass
        self.cap.release()


# ============================================================
# CONFIG
# ============================================================
CONNECTION_URL = "udp://:14551"

STREAM_URL = (
    "udp://0.0.0.0:2031"
    "?overrun_nonfatal=1"
    "&fifo_size=500000"
    "&flags=low_delay"
)

WINDOW_NAME = "FPV + AprilTag IBVS"

# Takeoff profile
CLIMB_SPEED = 2.0   # m/s (slightly gentler)
CLIMB_TIME = 5.0    # seconds

# Manual keyboard speeds (slower)
MANUAL_VEL_XY = 0.4
MANUAL_VEL_Z  = 0.4
YAW_RATE_DEG  = 25.0

SEND_HZ = 20
SEND_DT = 1.0 / SEND_HZ
DEADMAN_TIMEOUT = 1.5   # seconds

# IBVS speed limits (slow & safe)
MAX_VX_IBVS = 0.20   # forward/back
MAX_VY_IBVS = 0.20   # left/right
MAX_VZ_IBVS = 0.15   # up/down magnitude (Z+ is down)

# IBVS PID gains (slower/smoother)
PID_X = PID(kp=0.0005, ki=0.0, kd=0.00025)  # distance (area) -> X
PID_Y = PID(kp=0.0015, ki=0.0, kd=0.0007)   # dx -> Y
PID_Z = PID(kp=0.0015, ki=0.0, kd=0.0007)   # dy -> Z

# ---- AprilTag / Distance logic ----
# Tag physical size: 17 cm x 17 cm  (0.17 m)
TAG_SIZE_M = 0.17

# These area thresholds are approximate and MUST be tuned from real flight.
# Start conservative:
# - Stop approaching when tag_area >= AREA_STOP (≈ ~1.5 m)
# - Hard safety: if tag_area >= AREA_HARD_LIMIT -> no motion
AREA_STOP       = 4000.0    # initial guess for ~1.5 m (tune after flight)
AREA_HARD_LIMIT = 8000.0    # safety cap, do not get closer than this


# ============================================================
# SHARED STATE (PX4 body frame: X fwd, Y right, Z down)
# ============================================================
vx = 0.0
vy = 0.0
vz = 0.0
yaw_rate = 0.0

running = True
last_key_time = time.time()
takeoff_request = False
land_request = False
IBVS_enabled = False

state_lock = threading.Lock()

tag_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0
)


# ============================================================
# MAVSDK WORKER
# ============================================================
async def mavsdk_worker():
    global vx, vy, vz, yaw_rate
    global running, last_key_time
    global takeoff_request, land_request

    drone = System()
    print(f"[MAVSDK] Connecting to {CONNECTION_URL}")
    await drone.connect(system_address=CONNECTION_URL)

    # Wait for connection
    async for conn in drone.core.connection_state():
        if conn.is_connected:
            print("[MAVSDK] Connected.")
            break

    print("[MAVSDK] Waiting for GPS + home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[MAVSDK] Position OK.")
            break

    # PX4 status messages
    async def px4_msgs():
        async for msg in drone.telemetry.status_text():
            print(f"[PX4] {msg.type}: {msg.text}")

    asyncio.create_task(px4_msgs())

    offboard_started = False

    async def start_offboard():
        nonlocal offboard_started
        if offboard_started:
            return True
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
        except OffboardError as e:
            print("[MAVSDK] Initial OFFBOARD setpoint ERROR:", e._result)
            return False
        try:
            await drone.offboard.start()
            offboard_started = True
            print("[MAVSDK] OFFBOARD started.")
            return True
        except OffboardError as e:
            print("[MAVSDK] OFFBOARD start ERROR:", e._result)
            return False

    async def offboard_takeoff():
        nonlocal offboard_started
        print("[MAVSDK] Takeoff requested...")
        try:
            await drone.action.arm()
            if not await start_offboard():
                print("[MAVSDK] Takeoff aborted because OFFBOARD did not start.")
                return

            print("[MAVSDK] Climbing...")
            t0 = time.time()
            while time.time() - t0 < CLIMB_TIME and running:
                # Z+ = down; to go UP we use negative Z
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, -CLIMB_SPEED, 0.0)
                )
                await asyncio.sleep(SEND_DT)

            print("[MAVSDK] Hover phase...")
            for _ in range(40):
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                await asyncio.sleep(SEND_DT)

            print("[MAVSDK] Takeoff complete. Hovering.")
        except Exception as e:
            print("[MAVSDK] Takeoff ERROR:", e)

    # MAIN LOOP
    while running:
        now = time.time()

        with state_lock:
            lx = vx
            ly = vy
            lz = vz
            lyaw = yaw_rate

            do_takeoff = takeoff_request
            do_land   = land_request

            takeoff_request = False
            land_request = False

            last_input = last_key_time

        # Takeoff
        if do_takeoff:
            await offboard_takeoff()

        # Land
        if do_land:
            print("[MAVSDK] Landing requested...")
            try:
                if offboard_started:
                    try:
                        await drone.offboard.stop()
                    except OffboardError as e:
                        print("[MAVSDK] OFFBOARD stop before land ERROR:", e._result)
                    offboard_started = False
                await drone.action.land()
            except Exception as e:
                print("[MAVSDK] Land ERROR:", e)

        # Deadman timeout
        if now - last_input > DEADMAN_TIMEOUT:
            lx = ly = lz = lyaw = 0.0
            with state_lock:
                vx = vy = vz = yaw_rate = 0.0

        # Send OFFBOARD body velocity
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(lx, ly, lz, lyaw)
            )
        except OffboardError:
            # Ignore if OFFBOARD not yet started; takeoff will start it
            pass

        await asyncio.sleep(SEND_DT)

    # Cleanup
    print("[MAVSDK] Shutting down...")
    try:
        await drone.offboard.stop()
    except:
        pass
    try:
        await drone.action.disarm()
    except:
        pass


def start_mavsdk_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(mavsdk_worker())
    loop.close()


# ============================================================
# MAIN LOOP (FPV + APRILTAG + IBVS)
# ============================================================
def main():
    global vx, vy, vz, yaw_rate
    global takeoff_request, land_request
    global running, last_key_time, IBVS_enabled

    # Start MAVSDK thread
    t = threading.Thread(target=start_mavsdk_thread, daemon=True)
    t.start()

    # Start FPV camera
    camera = FastCamera(STREAM_URL)
    time.sleep(0.5)

    if not camera.ret:
        print("❌ No FPV stream available.")
        return

    print("✅ FPV stream running.\n")
    print("Controls:")
    print("  T = takeoff")
    print("  L = land")
    print("  F = toggle IBVS")
    print("  W/S = forward/back")
    print("  A/D = left/right")
    print("  ↑/↓ = up/down")
    print("  ESC = quit\n")

    while running:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        # Camera is physically flipped → fix with 180° rotation
        frame = cv2.flip(frame, -1)

        h, w = frame.shape[:2]
        cx = w // 2
        cy = h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = tag_detector.detect(gray)

        tag_found = False
        dx = dy = 0.0
        tag_area = 0.0

        # Draw camera center
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        if tags:
            tag_found = True
            tag = tags[0]  # first detection

            tx, ty = tag.center.astype(int)
            dx = float(tx - cx)  # +dx: tag to the RIGHT
            dy = float(ty - cy)  # +dy: tag BELOW center

            corners = tag.corners.astype(int)
            tag_area = float(cv2.contourArea(corners))

            # Draw tag & line to center
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
            cv2.line(frame, (cx, cy), (tx, ty), (255, 0, 0), 2)

        # ====================================================
        # IBVS CONTROL
        # ====================================================
        if IBVS_enabled:
            dt = SEND_DT

            if tag_found:
                # ---- SAFETY: DISTANCE LIMITS ----
                if tag_area >= AREA_HARD_LIMIT:
                    # Too close -> freeze motion
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                    vz_cmd = 0.0
                else:
                    # Horizontal alignment (RIGHT/LEFT)
                    # dx > 0 (tag right)  -> move right (Y+ => vy > 0)
                    # dx < 0 (tag left)   -> move left (Y- => vy < 0)
                    vy_cmd = PID_Y.update(dx, dt)   # <- FIXED: no extra minus

                    # Vertical alignment (UP/DOWN)
                    # dy > 0 (tag below)  -> move DOWN (Z+ => vz > 0)
                    # dy < 0 (tag above)  -> move UP   (Z- => vz < 0)
                    vz_cmd = PID_Z.update(dy, dt)

                    # Distance control via area:
                    # If area < AREA_STOP -> move forward (X+)
                    # If area >= AREA_STOP -> hold X (no more approach)
                    if tag_area < AREA_STOP:
                        area_error = AREA_STOP - tag_area
                        vx_cmd = PID_X.update(area_error, dt)
                    else:
                        vx_cmd = 0.0
                        PID_X.reset()

                    # Clamp IBVS speeds
                    vx_cmd = max(min(vx_cmd, MAX_VX_IBVS), -MAX_VX_IBVS)
                    vy_cmd = max(min(vy_cmd, MAX_VY_IBVS), -MAX_VY_IBVS)
                    vz_cmd = max(min(vz_cmd, MAX_VZ_IBVS), -MAX_VZ_IBVS)

                with state_lock:
                    vx = vx_cmd
                    vy = vy_cmd
                    vz = vz_cmd
                    last_key_time = time.time()
            else:
                # IBVS ON but no tag: hover (no motion)
                with state_lock:
                    vx = vy = vz = 0.0
                    last_key_time = time.time()

        # ====================================================
        # HUD
        # ====================================================
        cv2.putText(frame, f"IBVS: {IBVS_enabled}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if tag_found:
            cv2.putText(frame, f"dx: {dx:+.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255),2)
            cv2.putText(frame, f"dy: {dy:+.0f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255),2)
            cv2.putText(frame, f"Area: {int(tag_area)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255),2)
            cv2.putText(frame, f"STOP@≈1.5m Area≈{int(AREA_STOP)}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        else:
            cv2.putText(frame, "No AprilTag", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        with state_lock:
            vel_text = f"vx:{vx:+.2f} vy:{vy:+.2f} vz:{vz:+.2f}"
        cv2.putText(frame, vel_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # ====================================================
        # KEYBOARD HANDLING
        # ====================================================
        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            with state_lock:
                last_key_time = time.time()

        if key == ord('f'):
            with state_lock:
                IBVS_enabled = not IBVS_enabled
            PID_X.reset(); PID_Y.reset(); PID_Z.reset()
            print("IBVS =", IBVS_enabled)

        elif key == ord('t'):
            with state_lock:
                takeoff_request = True

        elif key == ord('l'):
            with state_lock:
                land_request = True

        # Manual override (disable IBVS)
        elif key == ord('w'):
            with state_lock:
                IBVS_enabled = False
                vx = MANUAL_VEL_XY
        elif key == ord('s'):
            with state_lock:
                IBVS_enabled = False
                vx = -MANUAL_VEL_XY
        elif key == ord('a'):
            with state_lock:
                IBVS_enabled = False
                vy = -MANUAL_VEL_XY
        elif key == ord('d'):
            with state_lock:
                IBVS_enabled = False
                vy = MANUAL_VEL_XY
        elif key == 82:  # up arrow -> UP (Z-)
            with state_lock:
                IBVS_enabled = False
                vz = -MANUAL_VEL_Z
        elif key == 84:  # down arrow -> DOWN (Z+)
            with state_lock:
                IBVS_enabled = False
                vz = MANUAL_VEL_Z
        elif key == 27:  # ESC
            with state_lock:
                running = False
            break

    camera.stop()
    cv2.destroyAllWindows()
    t.join(timeout=2.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MAVSDK PX4 FPV Keyboard Controller (OFFBOARD Body Velocity)
-----------------------------------------------------------
- OFFBOARD-only takeoff (same pattern as official MAVSDK example)
- Keyboard body velocity control
- Hover when no movement input (DEADMAN)
"""

import asyncio
import threading
import time

import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError


# ============================================================
# CONFIG
# ============================================================
CONNECTION_URL = "udp://:14551"   # keep same working URL

STREAM_URL = (
    "udp://0.0.0.0:2032"
    "?overrun_nonfatal=1"
    "&fifo_size=500000"
    "&flags=low_delay"
)

WINDOW_NAME = "FPV Stream"

# *** STRONGER TAKEOFF ***
CLIMB_SPEED = 2.5       # m/s upward (was 0.7)
CLIMB_TIME = 5.0        # seconds (was 2.5)

# *** STRONGER MANUAL COMMANDS ***
VEL = 1.2               # m/s XY/Z keyboard speed (was 0.5)
YAW_RATE_DEG = 30       # deg/s

SEND_HZ = 20
SEND_DT = 1.0 / SEND_HZ

DEADMAN_TIMEOUT = 0.5

# For now, don't zero velocities every frame with no key
# so we don't fight takeoff / motion
IMMEDIATE_RESET_ON_KEY_RELEASE = False


# ============================================================
# SHARED STATE
# ============================================================
vx = vy = vz = 0.0       # vz > 0 = UP
yaw_rate = 0.0

running = True
last_key_time = time.time()

takeoff_request = False
land_request = False

state_lock = threading.Lock()


# ============================================================
# MAVSDK WORKER
# ============================================================
async def mavsdk_worker():
    global running, vx, vy, vz, yaw_rate
    global last_key_time, takeoff_request, land_request

    drone = System()
    print(f"[MAVSDK] Connecting to PX4 at {CONNECTION_URL}")
    await drone.connect(system_address=CONNECTION_URL)

    # Wait for connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAVSDK] Connected.")
            break

    # REQUIRED FOR OFFBOARD: wait for global + home position
    print("[MAVSDK] Waiting for global position & home...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[MAVSDK] Position OK.")
            break

    # Print PX4 messages
    async def status_text_printer():
        async for status in drone.telemetry.status_text():
            print(f"[PX4] {status.type}: {status.text}")

    asyncio.create_task(status_text_printer())

    offboard_started = False

    async def start_offboard():
        """Initial 0,0,0,0 setpoint → OFFBOARD start (MAVSDK pattern)."""
        nonlocal offboard_started
        if offboard_started:
            return

        print("[MAVSDK] Sending initial OFFBOARD setpoint...")
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0, 0, 0, 0)
            )
        except OffboardError as e:
            print("[MAVSDK] Initial setpoint error:", e._result)

        print("[MAVSDK] Starting OFFBOARD...")
        try:
            await drone.offboard.start()
            offboard_started = True
            print("[MAVSDK] OFFBOARD started.")
        except OffboardError as e:
            print("[MAVSDK] OFFBOARD start FAILED:", e._result)
            return

    async def offboard_takeoff():
        """OFFBOARD-only vertical climb."""
        nonlocal offboard_started
        print("[MAVSDK] Takeoff requested.")

        try:
            print("[MAVSDK] Arming...")
            await drone.action.arm()

            # Start OFFBOARD
            await start_offboard()
            if not offboard_started:
                print("[MAVSDK] Cannot take off — OFFBOARD didn't start.")
                return

            # Climb using OFFBOARD velocity (Vz negative = up in PX4)
            print(f"[MAVSDK] Climbing {CLIMB_SPEED} m/s for {CLIMB_TIME}s...")
            t0 = time.time()
            while time.time() - t0 < CLIMB_TIME and running:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(
                        0.0,
                        0.0,
                        -CLIMB_SPEED,    # Vz negative = up (PX4 convention)
                        0.0
                    )
                )
                await asyncio.sleep(SEND_DT)

            # Stabilize hover
            print("[MAVSDK] Hover stabilization...")
            for _ in range(20):
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0, 0, 0, 0)
                )
                await asyncio.sleep(SEND_DT)

            print("[MAVSDK] Takeoff complete. Hovering.")
        except Exception as e:
            print("[MAVSDK] Takeoff error:", e)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    while running:
        now = time.time()

        # COPY shared state safely
        with state_lock:
            vx_local = vx
            vy_local = vy
            vz_local = vz
            yaw_local = yaw_rate

            takeoff_flag = takeoff_request
            land_flag = land_request

            takeoff_request = False
            land_request = False

            last_input = last_key_time

        # TAKEOFF
        if takeoff_flag and not offboard_started:
            await offboard_takeoff()

        # LAND
        if land_flag:
            print("[MAVSDK] Landing requested.")
            try:
                if offboard_started:
                    try:
                        await drone.offboard.stop()
                    except OffboardError:
                        pass
                offboard_started = False
                await drone.action.land()
                print("[MAVSDK] Land command sent.")
            except Exception as e:
                print("[MAVSDK] Land error:", e)

        # DEADMAN timeout (still keep for safety, but slower than per-frame reset)
        if now - last_input > DEADMAN_TIMEOUT:
            vx_local = vy_local = vz_local = yaw_local = 0.0
            with state_lock:
                vx = vy = vz = yaw_rate = 0.0

        # SEND OFFBOARD VELOCITY
        if offboard_started:
            try:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(
                        vx_local,
                        vy_local,
                        -vz_local,   # convert UP-positive to PX4 DOWN-positive
                        yaw_local
                    )
                )
            except OffboardError as e:
                print("[MAVSDK] Velocity error:", e._result)
                offboard_started = False

        await asyncio.sleep(SEND_DT)

    # Cleanup
    print("[MAVSDK] Worker exit.")
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
# MAIN THREAD (camera + keyboard)
# ============================================================
def main():
    global vx, vy, vz, yaw_rate, last_key_time
    global running, takeoff_request, land_request

    # Start MAVSDK in background thread
    t = threading.Thread(target=start_mavsdk_thread, daemon=True)
    t.start()

    # Camera
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("⚠ FPV stream failed. Using dummy frame.")
        cap = None

    print("\nKEYBOARD CONTROLS:")
    print("  T = OFFBOARD-only takeoff")
    print("  L = Land")
    print("  W/S = forward/back")
    print("  A/D = left/right")
    print("  ↑/↓ = up/down")
    print("  ←/→ = yaw left/right")
    print("  ESC = exit\n")

    while running:
        # FRAME READ
        if cap:
            ret, frame = cap.read()
            if not ret:
                continue
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Feed", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # HUD
        with state_lock:
            hud = f"vx:{vx:+.2f} vy:{vy:+.2f} vz:{vz:+.2f} yaw:{yaw_rate:+.2f}"

        cv2.putText(frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        movement = False

        if key != 255:
            with state_lock:
                last_key_time = time.time()

        # MOVEMENT KEYS
        if key == ord('w'):
            with state_lock: vx = VEL
            movement = True
        elif key == ord('s'):
            with state_lock: vx = -VEL
            movement = True
        elif key == ord('a'):
            with state_lock: vy = -VEL
            movement = True
        elif key == ord('d'):
            with state_lock: vy = VEL
            movement = True
        elif key == 82:   # up arrow
            with state_lock: vz = VEL
            movement = True
        elif key == 84:   # down arrow
            with state_lock: vz = -VEL
            movement = True
        elif key == 81:   # left arrow
            with state_lock: yaw_rate = -YAW_RATE_DEG
            movement = True
        elif key == 83:   # right arrow
            with state_lock: yaw_rate = YAW_RATE_DEG
            movement = True

        # ACTION KEYS
        elif key == ord('t'):
            with state_lock: takeoff_request = True
        elif key == ord('l'):
            with state_lock: land_request = True
        elif key == 27:
            running = False
            break

        # (Deadman per-frame reset is disabled above for now)

    # CLEANUP
    if cap: cap.release()
    cv2.destroyAllWindows()
    t.join(timeout=2)
    print("Exited.")


if __name__ == "__main__":
    main()

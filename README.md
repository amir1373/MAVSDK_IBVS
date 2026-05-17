# MAVSDK IBVS

Python experiments for MAVSDK-based drone control and image-based visual servoing (IBVS), including AprilTag detection and camera test scripts.

## What This Repository Contains

- `MAVSDK.py` - MAVSDK control experiment.
- `MAVSDK IBVS1.py` - MAVSDK/IBVS experiment.
- `AprilTagDetectionAlone.py` - standalone AprilTag detection test.
- `AprilTagDetectionDroneCamerUpsideDown.py` - AprilTag detection for an inverted drone-camera setup.
- `camTest.py` - camera test script.
- `CameraCode.txt` and `deadmanSwitch.txt` - notes/snippets related to camera and safety-control behavior.
- `.github/workflows/python-compile.yml` - lightweight Python compile check.

## Suggested Workflow

1. Verify the camera feed and calibration.
2. Run AprilTag detection independently.
3. Confirm MAVSDK connection to PX4 or simulation.
4. Test visual-servoing behavior in simulation before hardware.

## Safety

Drone-control scripts should be tested in simulation or with props removed before real flight. Verify failsafe behavior, frame conventions, camera orientation, and MAVSDK connection settings before running against hardware.
# MAVSDK IBVS

Python experiments for MAVSDK-based drone control and image-based visual servoing (IBVS), including AprilTag detection and camera test scripts.

## Contents

- `MAVSDK.py` - MAVSDK control experiment.
- `MAVSDK IBVS1.py` - IBVS-related MAVSDK experiment.
- `AprilTagDetectionAlone.py` - standalone AprilTag detection test.
- `AprilTagDetectionDroneCamerUpsideDown.py` - AprilTag detection variant for an inverted drone camera setup.
- `camTest.py` - camera test script.
- `CameraCode.txt` and `deadmanSwitch.txt` - notes/snippets for camera and safety-control behavior.

## Safety

Drone-control scripts should be tested in simulation or with props removed before real flight. Verify failsafe behavior, frame conventions, camera orientation, and MAVSDK connection settings before running against hardware.

## Notes

This is an experimental robotics repository. The scripts may require local camera hardware, MAVSDK/PX4 setup, and environment-specific connection strings.
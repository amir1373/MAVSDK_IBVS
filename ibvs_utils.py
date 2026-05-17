"""Small helpers for image-based visual servoing experiments.

These functions keep the math and safety clamping separate from drone-control
scripts, making it easier to test IBVS behavior before connecting to MAVSDK.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePoint:
    x: float
    y: float


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float


def normalized_error(target: ImagePoint, measured: ImagePoint, width: float, height: float) -> tuple[float, float]:
    """Return target-measured image error normalized to [-1, 1] scale."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    ex = 2.0 * (target.x - measured.x) / width
    ey = 2.0 * (target.y - measured.y) / height
    return ex, ey


def clamp(value: float, limit: float) -> float:
    limit = abs(limit)
    return max(-limit, min(limit, value))


def proportional_ibvs_command(
    error_x: float,
    error_y: float,
    *,
    gain_xy: float = 0.4,
    gain_yaw: float = 0.3,
    max_xy: float = 0.5,
    max_yaw: float = 0.4,
) -> VelocityCommand:
    """Convert normalized image error into a conservative body-frame command."""
    return VelocityCommand(
        vx=0.0,
        vy=clamp(gain_xy * error_x, max_xy),
        vz=clamp(-gain_xy * error_y, max_xy),
        yaw_rate=clamp(gain_yaw * error_x, max_yaw),
    )
from ibvs_utils import ImagePoint, normalized_error, proportional_ibvs_command


def test_normalized_error_centered():
    assert normalized_error(ImagePoint(320, 240), ImagePoint(320, 240), 640, 480) == (0.0, 0.0)


def test_command_is_clamped():
    command = proportional_ibvs_command(10.0, -10.0, max_xy=0.2, max_yaw=0.1)
    assert command.vy == 0.2
    assert command.vz == 0.2
    assert command.yaw_rate == 0.1
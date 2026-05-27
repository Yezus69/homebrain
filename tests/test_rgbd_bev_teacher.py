import numpy as np

from homebrain.teachers.rgbd_bev_teacher import RGBDBEVTeacherConfig, build_rgbd_bev_labels


def test_rgbd_bev_teacher_projects_depth_and_marks_dynamic_residual() -> None:
    config = RGBDBEVTeacherConfig(grid_shape=(16, 16), meters_per_cell=0.1, pixel_stride=1)
    intrinsics = {"fx": 4.0, "fy": 4.0, "cx": 1.5, "cy": 1.5, "depth_scale": 1000.0}
    depth_a = np.full((4, 4), 0.8, dtype=np.float32)
    depth_b = depth_a.copy()
    depth_b[:, 2:] = 1.2

    first = build_rgbd_bev_labels(depth_m=depth_a, intrinsics=intrinsics, config=config)
    second = build_rgbd_bev_labels(
        depth_m=depth_b,
        intrinsics=intrinsics,
        config=config,
        previous_labels=first,
        pose_delta_prev=(0.0, 0.0, 0.0),
    )

    assert first.metadata["weak_label"] is True
    assert first.metadata["product_training_approved"] is False
    assert first.current_bev_free.shape == (16, 16)
    assert np.count_nonzero(first.current_bev_occupied) > 0
    assert np.count_nonzero(first.current_bev_unknown) > 0
    assert np.count_nonzero(second.dynamic_residual_risk) > 0
    assert np.max(second.current_bev_risky) == 1.0

import numpy as np

from homebrain.brain.direct_rgbd_features import rgbd_patch_features


def test_rgbd_patch_features_regular_grid_uses_real_rgbd_statistics() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:2, :2, 0] = 255
    rgb[:2, 2:, 1] = 128
    rgb[2:, :2, 2] = 64
    depth = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.5, 0.5, 6.0, 6.0],
            [0.0, 0.5, 6.0, 6.0],
        ],
        dtype=np.float32,
    )

    features = rgbd_patch_features(
        rgb=rgb,
        depth_m=depth,
        feature_dim=10,
        patch_shape=(2, 2),
    )

    assert features.shape == (2, 2, 10)
    assert features[0, 0, 0] == 1.0
    assert features[0, 1, 1] == np.float32(128.0 / 255.0)
    assert features[1, 0, 2] == np.float32(64.0 / 255.0)
    assert features[0, 0, 3] == np.float32(2.5 / 6.0)
    assert features[0, 0, 7] == 1.0
    assert features[0, 1, 7] == 0.0
    assert features[1, 0, 7] == 0.75
    assert features[1, 1, 3] == 1.0

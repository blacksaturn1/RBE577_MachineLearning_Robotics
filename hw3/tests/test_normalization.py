import numpy as np
import math

from hw3.src.dubins_train import DubinsDataset


def test_traj_and_cond_roundtrip():
    # Create a lightweight DubinsDataset instance without running __init__
    ds = object.__new__(DubinsDataset)
    ds.normalize = True
    ds.MAX_GRID = 500

    # Prepare a small trajectory and check denormalize_traj inverts the min/max mapping
    orig_traj = np.array([[-500.0, 0.0, 10.0], [250.0, -250.0, 100.0]], dtype=np.float32)
    max_v = float(ds.MAX_GRID)
    min_v = -max_v
    denom = max_v - min_v if (max_v - min_v) != 0.0 else 1.0

    normalized_traj = (orig_traj - min_v) / denom
    denorm_traj = ds.denormalize_traj(normalized_traj)
    np.testing.assert_allclose(denorm_traj, orig_traj, atol=1e-6)

    # Prepare a condition vector: [x1,y1,x2,y2,yaw,gamma,sin(yaw),cos(yaw)]
    x1, y1, x2, y2 = -500.0, 0.0, 250.0, -250.0
    yaw = -1.0  # negative yaw (radians) to test wrapping
    gamma = -0.5
    sy = float(math.sin(yaw))
    cy = float(math.cos(yaw))

    # Forward normalization as implemented in __getitem__ (positions -> [0,1], yaw/gamma wrapped -> [0,1))
    pos = np.array([x1, y1, x2, y2], dtype=np.float32)
    pos_norm = (pos - min_v) / denom

    yaw_wrapped = (float(yaw) + 2.0 * math.pi) % (2.0 * math.pi)
    yaw_norm = yaw_wrapped / (2.0 * math.pi)

    gamma_wrapped = (float(gamma) + 2.0 * math.pi) % (2.0 * math.pi)
    gamma_norm = gamma_wrapped / (2.0 * math.pi)

    cond_norm = np.concatenate([pos_norm, np.array([yaw_norm, gamma_norm, sy, cy], dtype=np.float32)])

    denorm_cond = ds.denormalize_cond(cond_norm)

    # denormalize_cond returns positions in original scale and yaw/gamma multiplied by 2*pi
    assert denorm_cond.shape == (8,)
    np.testing.assert_allclose(denorm_cond[:4], pos, atol=1e-6)
    # yaw/gamma returned will be in [0,2*pi) due to wrapping performed in forward normalization
    np.testing.assert_allclose(denorm_cond[4], yaw_wrapped, atol=1e-6)
    np.testing.assert_allclose(denorm_cond[5], gamma_wrapped, atol=1e-6)
    # sin/cos are left unchanged by normalization/denormalization
    np.testing.assert_allclose(denorm_cond[6], sy, atol=1e-6)
    np.testing.assert_allclose(denorm_cond[7], cy, atol=1e-6)

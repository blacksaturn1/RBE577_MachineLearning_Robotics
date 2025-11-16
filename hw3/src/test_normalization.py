import os
import numpy as np
from hw3.src.dubins_train import DubinsDataset


def test_normalization_stats_and_denorm(tmp_path):
    # Prepare tiny on-disk dataset (two samples) inside tmp_path
    data_root = str(tmp_path)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    # Sample 0: 3 points near origin
    traj0 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    cond0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Sample 1: 2 points far from origin
    traj1 = np.array([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]], dtype=np.float32)
    cond1 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    archive_path = samples_dir / "all_samples_part000.npz"
    np.savez_compressed(
        archive_path,
        **{
            "traj_0": traj0,
            "cond_0": cond0,
            "yaw_0": np.float32(0.0),
            "gamma_0": np.float32(0.0),
            "quadrant_0": np.int32(0),
            "traj_1": traj1,
            "cond_1": cond1,
            "yaw_1": np.float32(0.0),
            "gamma_1": np.float32(0.0),
            "quadrant_1": np.int32(0),
        },
    )

    # index pointing to the archive entries
    index = np.array(
        [
            {"filename": os.path.join("samples", archive_path.name), "quadrant": 0, "length": 3, "idx": 0},
            {"filename": os.path.join("samples", archive_path.name), "quadrant": 0, "length": 2, "idx": 1},
        ],
        dtype=object,
    )
    np.save(os.path.join(data_root, "index.npy"), index)

    # Create dataset with normalization enabled (will compute stats and cache them)
    ds = DubinsDataset(regenerate=False, data_root=data_root, normalize=True)

    # Expected trajectory statistics (manual calculation):
    # All trajectory points across both samples: [[0,0,0],[1,1,1],[2,2,2],[10,10,10],[12,12,12]]
    # mean per-dim = (0+1+2+10+12)/5 = 25/5 = 5.0
    np.testing.assert_allclose(ds.traj_mean, np.array([5.0, 5.0, 5.0], dtype=np.float32), rtol=1e-6)

    # cond mean now includes sin(yaw), cos(yaw), gamma
    # For our two samples (yaw=0 -> sin=0, cos=1), cond_full are:
    # sample0: [0,0,0,0, 0,1,0]
    # sample1: [2,2,2,2, 0,1,0]
    # mean = [1,1,1,1, 0,1,0]
    np.testing.assert_allclose(ds.cond_mean, np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32), rtol=1e-6)

    # Check cond std: first 4 dims are 1.0 (0 and 2 -> std=1). sin/cos/gamma have zero variance -> clipped to norm_eps
    expected_cond_std = np.array([1.0, 1.0, 1.0, 1.0, ds.norm_eps, ds.norm_eps, ds.norm_eps], dtype=np.float32)
    np.testing.assert_allclose(ds.cond_std, expected_cond_std, rtol=1e-6)

    # Trajectory std (per-dim) approximate check
    # Compute expected variance manually for the five points
    pts = np.vstack([traj0, traj1])
    expected_std = np.sqrt(np.maximum(np.mean(pts.astype(np.float64) ** 2, axis=0) - (np.mean(pts, axis=0) ** 2), 0.0)).astype(np.float32)
    np.testing.assert_allclose(ds.traj_std, expected_std, rtol=1e-5)

    # __getitem__ should return normalized traj and full condition (including yaw/gamma)
    item0 = ds[0]
    traj0_norm = item0["traj"]
    cond0_norm = item0["cond"]

    # build full cond for sample0 (sin=0, cos=1, gamma=0) and compute expected normalized
    cond0_full = np.concatenate([cond0, np.array([0.0, 1.0, 0.0], dtype=np.float32)])
    exp_traj0_norm = (traj0 - ds.traj_mean.reshape(1, -1)) / ds.traj_std.reshape(1, -1)
    exp_cond0_norm = (cond0_full - ds.cond_mean) / ds.cond_std

    np.testing.assert_allclose(traj0_norm, exp_traj0_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cond0_norm, exp_cond0_norm, rtol=1e-6, atol=1e-6)

    # denormalize helper should invert normalization
    denorm = ds.denormalize_traj(traj0_norm)
    np.testing.assert_allclose(denorm, traj0, rtol=1e-6, atol=1e-6)

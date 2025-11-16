import os
import numpy as np
import tempfile
from hw3.src.dubins_train import DubinsDataset


def test_denormalize_and_meta_cond_preference(tmp_path):
    # Setup data directory
    data_dir = tmp_path / "data"
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(parents=True)

    # Create a small trajectory
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)

    # Archive cond (different from meta cond to ensure preference)
    archive_cond = np.array([9.0, 9.0, 10.0, 10.0, 0.5, -0.1], dtype=np.float32)

    # Meta cond that we want dataset to prefer
    meta_cond = np.array([0.0, 0.0, 1.0, 1.0, 0.25, 0.125], dtype=np.float32)

    yaw = np.float32(meta_cond[4])
    gamma = np.float32(meta_cond[5])
    quadrant = 0

    # Save archive with different cond
    archive_path = samples_dir / "all_samples_part000.npz"
    np.savez_compressed(archive_path,
                        traj_0=traj,
                        cond_0=archive_cond,
                        yaw_0=np.float32(yaw),
                        gamma_0=np.float32(gamma),
                        quadrant_0=np.int32(quadrant))

    # Build index referencing the archive, but include meta 'cond' (the preferred one)
    index = np.array([{
        "filename": os.path.join("samples", "all_samples_part000.npz"),
        "idx": 0,
        "length": traj.shape[0],
        "quadrant": quadrant,
        "cond": meta_cond.tolist(),
    }], dtype=object)
    np.save(data_dir / "index.npy", index)

    # Create norm stats so denormalize_cond is deterministic
    # Build cond_full from meta (x1,y1,x2,y2,yaw,gamma,sin(yaw),cos(yaw))
    sy = float(np.sin(float(yaw)))
    cy = float(np.cos(float(yaw)))
    cond_full = np.concatenate([meta_cond, np.array([sy, cy], dtype=np.float32)])

    # Choose non-trivial mean/std
    cond_mean = cond_full * 0.5
    cond_std = cond_full * 0.75
    # Avoid zeros in std
    cond_std[cond_std == 0] = 1.0

    traj_mean = np.zeros(3, dtype=np.float32)
    traj_std = np.ones(3, dtype=np.float32)

    np.savez(data_dir / "norm_stats.npz", traj_mean=traj_mean, traj_std=traj_std, cond_mean=cond_mean.astype(np.float32), cond_std=cond_std.astype(np.float32))

    # Now load dataset and verify behaviors
    ds = DubinsDataset(regenerate=False, data_root=str(data_dir), normalize=True)

    # The dataset __getitem__ returns normalized cond_full (cond + sin/cos)
    item = ds[0]
    returned_cond = item["cond"]  # normalized (8,)

    # Expected normalized cond_full based on meta_cond
    expected_norm = (cond_full - cond_mean) / cond_std

    assert np.allclose(returned_cond, expected_norm, atol=1e-6), "Dataset did not prefer meta['cond'] when loading cond"

    # Test denormalize_cond roundtrip
    denorm = ds.denormalize_cond(returned_cond)
    assert np.allclose(denorm, cond_full, atol=1e-6), "denormalize_cond did not recover original cond_full"

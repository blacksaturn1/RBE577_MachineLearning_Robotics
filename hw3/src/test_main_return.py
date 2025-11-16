#!/usr/bin/env python3
"""Test that main() returns a consistent tuple and runs with small debug dataset."""
import os
import numpy as np

from dubins_train import main, DubinsDataset


def create_small_dataset(n_samples=8, out_dir='data'):
    samples_dir = os.path.join(out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    archive = {}
    index_list = []
    for k in range(n_samples):
        L = np.random.randint(3, 8)
        traj = np.random.randn(L, 3).astype(np.float32)
        cond = np.array([0.0, 0.0, float(k), float(k)], dtype=np.float32)
        yaw = np.float32(0.0)
        gamma = np.float32(0.0)
        quadrant = np.int32(k % 4)
        archive[f"traj_{k}"] = traj
        archive[f"cond_{k}"] = cond
        archive[f"yaw_{k}"] = yaw
        archive[f"gamma_{k}"] = gamma
        archive[f"quadrant_{k}"] = quadrant
        index_list.append({
            "filename": os.path.join('samples', 'all_samples_part000.npz'),
            "idx": k,
            "length": int(L),
            "quadrant": int(quadrant),
        })

    np.savez_compressed(os.path.join(samples_dir, 'all_samples_part000.npz'), **archive)
    np.save(os.path.join(out_dir, 'index.npy'), np.array(index_list, dtype=object))


def test_main_return_consistency(tmp_path):
    td = str(tmp_path)
    create_small_dataset(n_samples=8, out_dir=td)
    # run main with tiny settings, point data_root and model_dir at the temp dir
    model, dataset, train_loaders, val_loaders, test_loaders, history = main(
        batch_size=2,
        epochs=1,
        lr=1e-3,
        tf_ratio=0.0,
        regenerate_dataset=False,
        dynamic_batching=True,
        bucket_size=4,
        data_root=td,
        model_dir=td,
    )
    assert model is not None
    assert isinstance(history, dict)
    assert len(train_loaders) == 4 and len(val_loaders) == 4 and len(test_loaders) == 4

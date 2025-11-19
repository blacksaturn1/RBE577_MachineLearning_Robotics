#!/usr/bin/env python3
"""Test dynamic batching (BucketBatchSampler + DataLoader integration)."""
import os
import numpy as np
import torch

from dubins_train import DubinsDataset, build_quadrant_loaders


def create_small_dataset(n_samples=20, out_dir='data'):
    samples_dir = os.path.join(out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    archive = {}
    index_list = []
    for k in range(n_samples):
        L = np.random.randint(3, 12)
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


def test_dynamic_batching_loader_yields_batches(tmp_path):
    td = str(tmp_path)
    create_small_dataset(n_samples=20, out_dir=td)
    ds = DubinsDataset(regenerate=False, data_root=td)
    # dynamic batching enabled
    train_loaders, val_loaders, test_loaders, *_ = build_quadrant_loaders(ds, batch_size=4, val_split=0.2, test_split=0.1, dynamic_batching=True, bucket_size=8)

    # iterate one batch from each quadrant's train loader to ensure it yields
    for q in range(4):
        loader = train_loaders[q]
        if len(loader.dataset) == 0:
            continue
        it = iter(loader)
        batch = next(it)
        assert isinstance(batch, tuple) and len(batch) == 3, "Expected collate output (trajs,lengths,conds)"
        trajs, lengths, conds = batch
        assert trajs.dim() == 3
        assert lengths.dim() == 1
        assert conds.dim() == 2

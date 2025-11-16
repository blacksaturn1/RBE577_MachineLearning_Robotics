#!/usr/bin/env python3
"""
Create a tiny on-disk dataset (part archive + index), verify build_quadrant_loaders splits,
and run a quick training iteration across quadrants to validate the pipeline.
"""
import os
import numpy as np
import torch

from dubins_train import DubinsDataset, build_quadrant_loaders, DubinsLSTM, train_epoch


def create_small_dataset(n_samples=12, out_dir='data'):
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


def test_loader_splits_and_quadrant_training(tmp_path):
    # Prepare small dataset on disk under pytest tmp_path
    td = str(tmp_path)
    create_small_dataset(n_samples=12, out_dir=td)

    ds = DubinsDataset(regenerate=False, data_root=td)
    batch_size = 4
    # request 25% val, 25% test
    train_loaders, val_loaders, test_loaders, train_idxs, val_idxs, test_idxs = build_quadrant_loaders(
        ds, batch_size=batch_size, val_split=0.25, test_split=0.25, shuffle_index=False
    )

    n_total = sum(len(ld.dataset) for ld in train_loaders) + sum(len(ld.dataset) for ld in val_loaders) + sum(len(ld.dataset) for ld in test_loaders)
    assert n_total == 12, f"Expected 12 samples, got {n_total}"

    # run a small training iteration (1 epoch across quadrants) on CPU
    device = torch.device('cpu')
    model = DubinsLSTM().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss(reduction='none')

    # ensure training passes without errors and returns non-negative loss
    for q in range(4):
        loader = train_loaders[q]
        if len(loader.dataset) == 0:
            continue
        loss = train_epoch(model, loader, optim, device, criterion, teacher_forcing=0.0)
        assert isinstance(loss, float) and loss >= 0.0

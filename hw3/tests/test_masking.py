import numpy as np
import torch

from hw3.src.dubins_train import collate_fn


def test_masking_loss_and_metrics_on_toy_batch():
    """Create a tiny batch with variable-length trajectories and verify
    the masked loss, ADE and FDE calculations match hand-computed values.
    """
    # Sequence 1: length 3
    traj1 = np.array([[1.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [3.0, 0.0, 0.0]], dtype=np.float32)
    cond1 = np.zeros(8, dtype=np.float32)

    # Sequence 2: length 1
    traj2 = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    cond2 = np.zeros(8, dtype=np.float32)

    batch = [
        {"traj": traj1, "cond": cond1},
        {"traj": traj2, "cond": cond2},
    ]

    padded, lengths, conds = collate_fn(batch)

    # preds: zeros (same shape as padded)
    preds = torch.zeros_like(padded)

    # --- Loss as implemented in repo (elementwise MSE averaged over coords, masked over time) ---
    loss_elem = (preds - padded) ** 2
    loss_t = loss_elem.mean(dim=-1)  # mean over coordinate dim
    max_len = preds.size(1)
    device = preds.device
    mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    loss_masked = (loss_t * mask).sum() / mask.sum()

    # Manual expected values:
    # For traj1 timesteps squared-error per-timestep (pred=0): [1,4,9], mean over coords -> divide by 3
    # For traj2: [16] -> mean -> 16/3
    # Sum over valid timesteps = (1+4+9)/3 + (16)/3 = (30)/3 = 10
    # Total valid timesteps = 3 + 1 = 4 -> average = 10 / 4 = 2.5
    expected_loss = 2.5

    assert torch.isclose(loss_masked, torch.tensor(expected_loss, device=device), atol=1e-6)

    # --- ADE calculation (average L2 norm per sequence, then mean across sequences) ---
    norms = torch.norm(preds - padded, dim=-1)
    per_seq_ADE = (norms * mask).sum(dim=1) / lengths.float()
    avg_ADE = per_seq_ADE.sum().item() / len(batch)

    # Manual ADE: seq1 (1+2+3)/3 = 2.0; seq2 = 4/1 = 4.0 -> mean = (2 + 4) / 2 = 3.0
    assert abs(avg_ADE - 3.0) < 1e-6

    # --- FDE calculation (L2 norm at last valid timestep, averaged across batch) ---
    idx = (lengths - 1).clamp(min=0)
    batch_idx = torch.arange(preds.size(0), device=device)
    last_preds = preds[batch_idx, idx, :]
    last_gt = padded[batch_idx, idx, :]
    fde_per = torch.norm(last_preds - last_gt, dim=-1)
    avg_FDE = float(fde_per.sum().item()) / len(batch)

    # Manual FDE: seq1 last=3 -> 3.0; seq2 last=4 -> 4.0 -> mean = 3.5
    assert abs(avg_FDE - 3.5) < 1e-6

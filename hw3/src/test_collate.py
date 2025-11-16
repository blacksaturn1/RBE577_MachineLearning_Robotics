import sys
import os
import types
import numpy as np
import torch

# Prevent importing the real matplotlib (which depends on system numpy ABI) during this small unit test.
# Provide lightweight fake modules so importing `dubins_train` succeeds in this environment.
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib._path'] = types.ModuleType('matplotlib._path')
sys.modules['matplotlib.transforms'] = types.ModuleType('matplotlib.transforms')
sys.modules['matplotlib.colors'] = types.ModuleType('matplotlib.colors')
sys.modules['matplotlib.ticker'] = types.ModuleType('matplotlib.ticker')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import dubins_train as dt


def test_collate_and_forward_shapes_and_masked_loss():
    torch.manual_seed(0)
    np.random.seed(0)

    # create a fake batch with varying lengths
    lengths_list = [5, 8, 3]
    batch = []
    for L in lengths_list:
        traj = np.random.randn(L, 3).astype(np.float32)
        # include sin(yaw), cos(yaw) and gamma to match dataset cond format
        # here yaw==0 -> sin=0, cos=1, gamma=0
        cond = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        batch.append({'traj': traj, 'cond': cond})

    padded, lengths, conds = dt.collate_fn(batch)
    assert padded.shape[0] == 3 and padded.dim() == 3
    assert lengths.shape[0] == 3
    assert conds.shape == (3, 7)

    model = dt.DubinsLSTM()
    model.eval()
    with torch.no_grad():
        preds = model(conds, target_seq=padded, lengths=lengths, teacher_forcing_ratio=0.0)

    assert preds.shape == padded.shape

    # compute masked mse (same logic as training) and ensure it's finite
    loss_elem = (preds - padded) ** 2
    loss_t = loss_elem.mean(dim=-1)
    max_len = preds.size(1)
    mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).float()
    loss = (loss_t * mask).sum() / mask.sum()
    assert torch.isfinite(loss)

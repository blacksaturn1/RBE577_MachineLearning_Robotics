import sys
import os
import types
# Prevent importing the real matplotlib (which depends on system numpy ABI) during this small unit test.
# We provide lightweight fake modules so importing `dubins_train` succeeds in this environment.
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')
sys.modules['matplotlib._path'] = types.ModuleType('matplotlib._path')
sys.modules['matplotlib.transforms'] = types.ModuleType('matplotlib.transforms')
sys.modules['matplotlib.colors'] = types.ModuleType('matplotlib.colors')
sys.modules['matplotlib.ticker'] = types.ModuleType('matplotlib.ticker')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import dubins_train as dt
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# create a fake batch with varying lengths
lengths_list = [5, 8, 3]
batch = []
for L in lengths_list:
    traj = np.random.randn(L, 3).astype(np.float32)
    cond = np.array([0., 0., 1., 1.], dtype=np.float32)
    batch.append({'traj': traj, 'cond': cond})

padded, lengths, conds = dt.collate_fn(batch)
print('padded.shape=', padded.shape)
print('lengths=', lengths)
print('conds.shape=', conds.shape)

model = dt.DubinsLSTM()
model.eval()
with torch.no_grad():
    preds = model(conds, target_seq=padded, lengths=lengths, teacher_forcing_ratio=0.0)

print('preds.shape=', preds.shape)
# compute masked mse (same logic as training)
loss_elem = (preds - padded) ** 2
loss_t = loss_elem.mean(dim=-1)
max_len = preds.size(1)
mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).float()
loss = (loss_t * mask).sum() / mask.sum()
print('masked mse=', float(loss.item()))

print('Test completed successfully')

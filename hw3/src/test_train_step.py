import sys
import os
import types
# Mock matplotlib to avoid import-time dependency issues when importing dubins_train
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import dubins_train as dt
import numpy as np
import torch
from torch.utils.data import DataLoader

# build a single-batch dataset (list) with varying lengths
batch_size = 4
lengths_list = [5, 7, 3, 6]
items = []
for L in lengths_list:
    traj = np.random.randn(L, 3).astype(np.float32)
    cond = np.array([0., 0., 1., 1.], dtype=np.float32)
    items.append({'traj': traj, 'cond': cond})

loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=dt.collate_fn)

# create model, optimizer, criterion
device = torch.device('cpu')
model = dt.DubinsLSTM().to(device)
optim_obj = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss(reduction='none')

# run one training epoch (will process exactly one batch)
train_loss = dt.train_epoch(model, loader, optim_obj, device, criterion, teacher_forcing=0.0, clip_grad=2.0)
print('train_loss (one step) =', train_loss)
print('Done')

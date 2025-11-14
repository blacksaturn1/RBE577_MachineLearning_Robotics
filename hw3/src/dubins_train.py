import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import os

from dubinEHF3d import dubinEHF3d
# from tqdm import tqdm

# Dataset

class DubinsDataset(Dataset):
    def __init__(self, num_samples=2500, seq_len=5):
        self.data = []
        self.seq_len = seq_len
        for _ in range(num_samples):
            traj, cond = self._generate_sample(seq_len)
            self.data.append({"traj": traj, "cond": cond})

    def _generate_sample(self, seq_len):
        
        x1, y1, alt1 = 0, 0, 0  # Start at origin
        x2, y2 = 1000, 1000  # Default goal position
        for _ in range(100000):  # Try up to 100 times to find a valid path
            # Random goal x,y

            x2, y2 = np.random.uniform(-3000, 3000, size=(2,))
            # Random start heading
            start_yaw = np.random.uniform(0., 2.0*np.pi)
            gamma = np.random.uniform(-30, 30) * np.pi / 180  # Random Climb angle between -15 to 15 degrees
            # Path parameters
            step_length = 10  # Trajectory discretization step size
            r_min = 100  # Minimum turn radius  
            path, psi_end, num_path_points = dubinEHF3d(
                x1, y1, alt1, start_yaw, x2, y2, r_min, step_length, gamma
            )
            if num_path_points <= seq_len:
                break
        traj = path[:num_path_points, :]
        # # If trajectory is shorter than seq_len, pad with last point
        # if num_path_points < seq_len:
        #     pad_length = seq_len - num_path_points
        #     pad_points = np.tile(traj[-1, :], (pad_length, 1))
        #     traj = np.vstack([traj, pad_points])
        return traj.astype(np.float32), np.array([x1, y1, x2, y2], dtype=np.float32)
    
        # # Random goal x,y
        # x2, y2 = np.random.uniform(-3000, 3000, size=(2,))
        # # Random start heading
        # start_yaw = np.random.uniform(0., 2.0*np.pi)
        # gamma = np.random.uniform(-30, 30) * np.pi / 180  # Random Climb angle between -15 to 15 degrees
        # # Path parameters
        # step_length = 10  # Trajectory discretization step size
        # r_min = 100  # Minimum turn radius

        # path, psi_end, num_path_points = dubinEHF3d(
        #     x1, y1, alt1, start_yaw, x2, y2, r_min, step_length, gamma
        # )
        # traj = path[:num_path_points, :]
        # cond = np.array([x1, y1, x2, y2], dtype=np.float32)
        # return traj.astype(np.float32), cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate a batch of dataset items (each item is a dict with 'traj' and 'cond').

    Returns:
        padded_trajs: (B, L_max, 3) float tensor
        lengths: (B,) long tensor with true lengths
        conds: (B, 4) float tensor
    """
    # support either dict items or (traj, cond) tuples
    trajs = []
    conds = []
    for item in batch:
        if isinstance(item, dict):
            trajs.append(torch.from_numpy(item['traj']))
            conds.append(torch.from_numpy(item['cond']))
        else:
            # assume tuple (traj, cond)
            trajs.append(torch.from_numpy(item[0]))
            conds.append(torch.from_numpy(item[1]))

    lengths = torch.tensor([t.size(0) for t in trajs], dtype=torch.long)
    batch_size = len(trajs)
    max_len = int(lengths.max().item()) if batch_size > 0 else 0

    # feature dim
    feat = trajs[0].size(1) if trajs[0].dim() > 1 else 1
    padded = torch.zeros(batch_size, max_len, feat, dtype=trajs[0].dtype)
    for i, t in enumerate(trajs):
        L = t.size(0)
        padded[i, :L, :] = t

    conds = torch.stack(conds).float()
    padded = padded.float()
    return padded, lengths, conds

# Model

class DubinsLSTM(nn.Module):
    def __init__(self, input_dim=3, cond_dim=4, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_cond = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, conds, target_seq=None, lengths=None, teacher_forcing_ratio=0.5, seq_len: int = 50):
        """Autoregressive forward. If target_seq is provided, model runs for target_seq.size(1) steps
        and uses per-sample teacher forcing controlled by `lengths` so padded timesteps are not used.
        If target_seq is None, generate `seq_len` timesteps.

        Args:
            conds: (B, cond_dim)
            target_seq: (B, T, 3) or None
            lengths: (B,) int tensor with true lengths (required if using target_seq)
            teacher_forcing_ratio: probability to use teacher forcing for a timestep (per sample)
            seq_len: number of timesteps to generate when target_seq is None
        Returns:
            out: (B, T_out, 3)
        """
        B = conds.size(0)
        device = conds.device
        cond_embed = self.fc_cond(conds)
        cond_embed = cond_embed.unsqueeze(1)  # (B, 1, hidden_dim)
        # decide how many timesteps to run
        if target_seq is not None:
            run_len = target_seq.size(1)
        else:
            run_len = seq_len

        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        out_seq = []

        # initialize input with zeros
        prev_out = torch.zeros(B, 1, 3, device=device)
        for t in range(run_len):
            # rnn input is previous output concatenated with conditioning
            rnn_in = torch.cat([prev_out, cond_embed], dim=-1)
            out, (h, c) = self.lstm(rnn_in, (h, c))
            pred = self.fc_out(out)  # (B,1,3)
            out_seq.append(pred)

            # prepare prev_out for next step
            if target_seq is not None:
                # prefer teacher forcing only for sequences that haven't ended
                if lengths is None:
                    # fallback: use batch-level teacher forcing
                    use_tf = (random.random() < teacher_forcing_ratio)
                    if use_tf:
                        prev_out = target_seq[:, t:t+1, :]
                    else:
                        prev_out = pred
                else:
                    # per-sample decision: only apply teacher forcing for those with t < length
                    rand = torch.rand(B, device=device)
                    use_tf = (rand < teacher_forcing_ratio) & (t < lengths)
                    use_tf = use_tf.view(B, 1, 1)
                    prev_target = target_seq[:, t:t+1, :]
                    prev_out = torch.where(use_tf, prev_target, pred)
            else:
                prev_out = pred

        return torch.cat(out_seq, dim=1)

# Training and Evaluation

def ADE(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def FDE(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1))


def train_epoch(model, loader, optim, device, criterion, teacher_forcing=0.5, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    for trajs, lengths, conds in loader:
        trajs = trajs.float().to(device)
        lengths = lengths.to(device)
        conds = conds.float().to(device)

        optim.zero_grad()
        preds = model(conds, target_seq=trajs, lengths=lengths, teacher_forcing_ratio=teacher_forcing)

        # masked MSE: compute per-timestep MSE and average over real timesteps
        # criterion expected to be MSELoss(reduction='none') or we fallback to manual squared error
        if isinstance(criterion, nn.MSELoss) and criterion.reduction == 'none':
            loss_elem = criterion(preds, trajs)  # (B, L, 3)
        else:
            loss_elem = (preds - trajs) ** 2
        loss_t = loss_elem.mean(dim=-1)  # (B, L)
        max_len = preds.size(1)
        mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        loss = (loss_t * mask).sum() / mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()
        running_loss += float(loss.item()) * trajs.size(0)
    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    total_ADE = 0.0
    total_FDE = 0.0
    with torch.no_grad():
        for trajs, lengths, conds in loader:
            trajs = trajs.float().to(device)
            lengths = lengths.to(device)
            conds = conds.float().to(device)

            seq_len = int(lengths.max().item())
            preds = model(conds, target_seq=None, seq_len=seq_len, teacher_forcing_ratio=0.0)

            # masked MSE
            if isinstance(criterion, nn.MSELoss) and criterion.reduction == 'none':
                loss_elem = criterion(preds, trajs)
            else:
                loss_elem = (preds - trajs) ** 2
            loss_t = loss_elem.mean(dim=-1)
            max_len = preds.size(1)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
            loss = (loss_t * mask).sum() / mask.sum()
            running_loss += float(loss.item()) * trajs.size(0)

            # ADE: mean per-sample over real timesteps, then mean over batch
            norms = torch.norm(preds - trajs, dim=-1)  # (B, L)
            per_seq_ADE = (norms * mask).sum(dim=1) / lengths.float()
            total_ADE += float(per_seq_ADE.sum().item())

            # FDE: distance at last ground-truth timestep per sample
            idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(preds.size(0), device=device)
            last_preds = preds[batch_idx, idx, :]
            last_gt = trajs[batch_idx, idx, :]
            fde_per = torch.norm(last_preds - last_gt, dim=-1)
            total_FDE += float(fde_per.sum().item())
    n = len(loader.dataset)
    return running_loss / n, total_ADE / n, total_FDE / n


def plot_prediction_example(model, dataset, device=None, idx=0, seq_len=50):
    """Plot one example prediction.

    If device is None, infer the device from the model's parameters so tensors are
    moved to the same device as the model (prevents CPU/GPU mismatch errors).
    """
    model.eval()

    # Infer device from model if not given
    model_device = device if device is not None else next(model.parameters()).device

    trajs, lengths, conds = collate_fn([dataset[idx]])
    trajs = trajs.float().to(model_device)
    lengths = lengths.to(model_device)
    conds = conds.float().to(model_device)
    gen_len = int(lengths[0].item())
    with torch.no_grad():
        preds = model(conds, target_seq=None, seq_len=gen_len, teacher_forcing_ratio=0.0)

    preds = preds.cpu().numpy()[0]
    gt = trajs.cpu().numpy()[0][:gen_len]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:,0], gt[:,1], gt[:,2], 'b.-', label='gt')
    ax.plot(preds[:,0], preds[:,1], preds[:,2], 'r.-', label='pred')
    ax.scatter([0],[0],[0], c='k', marker='*', s=80, label='start')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Main Training Loop

def main_train(num_samples=2500, seq_len=50, batch_size=64, epochs=30, lr=1e-3, tf_ratio=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating {num_samples} samples...")
    dataset = DubinsDataset(num_samples=num_samples, seq_len=seq_len)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = DubinsLSTM().to(device)
    optim_obj = optim.Adam(model.parameters(), lr=lr)
    # use reduction='none' so we can mask padded timesteps manually
    criterion = nn.MSELoss(reduction='none')

    history = {'train_loss': [], 'val_loss': [], 'ADE': [], 'FDE': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optim_obj, device, criterion,
                                 teacher_forcing=tf_ratio, clip_grad=2.0)
        val_loss, val_ADE, val_FDE = eval_epoch(model, val_loader, device, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ADE'].append(val_ADE)
        history['FDE'].append(val_FDE)

        print(f"[{epoch+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| ADE: {val_ADE:.3f} | FDE: {val_FDE:.3f}")

    torch.save(model.state_dict(), "dubin_lstm.pth")
    print("Training complete. Model saved as 'dubin_lstm.pth'.")

    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    return model, dataset, train_loader, val_loader, history


if __name__ == "__main__":
    model, dataset, train_loader, val_loader, history = main_train()
    plot_prediction_example(model, dataset)
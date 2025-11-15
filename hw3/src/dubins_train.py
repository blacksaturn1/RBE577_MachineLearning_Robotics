#!/usr/bin/env python3
"""
Full script with quadrant-by-yaw (Option B) data loader that saves samples to disk
and loads trajectories on demand to reduce RAM usage.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dubinEHF3d import dubinEHF3d

# -----------------------------
# Dataset (on-disk, indexed)
# -----------------------------

class DubinsDataset(Dataset):
    """
    Generates or loads an on-disk dataset of Dubins-like 3D trajectories.
    Each sample is saved as 'data/samples/{idx:08d}.npz' with keys:
        'traj'   : (N, 3) float32
        'cond'   : (4,) float32  -> [x1, y1, x2, y2]
        'yaw'    : float32 (radians)
        'gamma'  : float32 (radians)
        'quadrant': int (0..3)
    Index file: 'data/index.npy' stores a list of small metadata dicts:
        [{'filename': 'samples/00000000.npz', 'quadrant':0, 'length':N}, ...]
    """
    MAX_GRID = 500          # grid range +/- MAX_GRID
    X_Y_SPACE = 10          # grid spacing
    MAX_YAW = 360           # degrees (we iterate 0..360 inclusive step)
    MAX_MIN_ANGLE = 30      # degrees (gamma range -30..30)
    YAW_STEP = 10           # degrees
    MAX_MIN_ANGLE_STEP = 5  # degrees

    DATA_ROOT = "data"
    SAMPLES_DIR = os.path.join(DATA_ROOT, "samples")
    INDEX_FILE = os.path.join(DATA_ROOT, "index.npy")

    def __init__(self, regenerate: bool = False, r_min=100, step_length=10):
        """
        If regenerate=True, existing index/samples will be overwritten by new generation.
        """
        super().__init__()
        self.r_min = r_min
        self.step_length = step_length

        os.makedirs(self.SAMPLES_DIR, exist_ok=True)

        if regenerate or (not os.path.exists(self.INDEX_FILE)):
            print("Generating dataset (this may take time)...")
            self._generate_samples()
        else:
            print(f"Loading dataset index from '{self.INDEX_FILE}'")
        # load index (list of metadata dicts)
        self.index = np.load(self.INDEX_FILE, allow_pickle=True).tolist()
        print(f"Dataset contains {len(self.index)} samples (index loaded).")

    # -------------------------
    # helper: quadrant from yaw
    # -------------------------
    def _get_quadrant(self, yaw_rad: float) -> int:
        yaw_deg = (yaw_rad * 180.0 / np.pi) % 360.0
        if 0 <= yaw_deg < 90:
            return 0
        elif 90 <= yaw_deg < 180:
            return 1
        elif 180 <= yaw_deg < 270:
            return 2
        else:
            return 3

    # -------------------------
    # Try generating a single trajectory
    # -------------------------
    def _try_generate_sample(self, x1, y1, alt1, x2, y2, yaw, gamma):
        """
        Attempt to generate a valid path using dubinEHF3d.
        Try a few times and return traj (np.ndarray) or None.
        """
        for _ in range(10):
            path, psi_end, num_path_points = dubinEHF3d(
                x1, y1, alt1, yaw, x2, y2, self.r_min, self.step_length, gamma
            )
            if path is None:
                continue
            if num_path_points >= 2:
                traj = path[:num_path_points, :].astype(np.float32)
                return traj
        return None

    # -------------------------
    # Generate samples to disk
    # -------------------------
    def _generate_samples(self):

        # skip generation if index exists
        if os.path.exists(self.INDEX_FILE):
            print("Index file already exists; skipping data generation.")
            return


        # # remove old samples/index if present
        # if os.path.exists(self.INDEX_FILE):
        #     print("Removing old index file...")
        #     os.remove(self.INDEX_FILE)
        # # optional: wipe sample dir
        
        # WARNING: this will delete existing sample files
        for fname in os.listdir(self.SAMPLES_DIR):
            path = os.path.join(self.SAMPLES_DIR, fname)
            try:
                os.remove(path)
            except Exception:
                pass
        x1, y1, alt1 = 0.0, 0.0, 0.0  # start
        idx = 0
        index_list = []
        # buffer for bulk saving; we flush to disk every FLUSH_EVERY samples to avoid high memory use
        archive_buf = {}
        part_idx = 0
        FLUSH_EVERY = 10  # flush samples
        current_part_name = f"all_samples_part{part_idx:03d}.npz"

        yaw_values = list(range(0, self.MAX_YAW, self.YAW_STEP))  # e.g., 0..350
        gamma_values = list(range(-self.MAX_MIN_ANGLE, self.MAX_MIN_ANGLE + 1, self.MAX_MIN_ANGLE_STEP))

        expected = ((2 * self.MAX_GRID) // self.X_Y_SPACE + 1) ** 2 * len(yaw_values) * len(gamma_values)
        print(f"Expected grid samples (upper bound): {expected}")
        debug = False
        debug_stopping_point = 10000
        # iterate deterministically (non-random)
        for x2 in range(-self.MAX_GRID, self.MAX_GRID + 1, self.X_Y_SPACE):
            if debug and idx > debug_stopping_point:
                break
            for y2 in range(-self.MAX_GRID, self.MAX_GRID + 1, self.X_Y_SPACE):
                if debug and idx > debug_stopping_point:
                    break
                for yaw_deg in yaw_values:
                    yaw_rad = yaw_deg * np.pi / 180.0
                    for gamma_deg in gamma_values:
                        gamma_rad = gamma_deg * np.pi / 180.0
                        traj = self._try_generate_sample(x1, y1, alt1, float(x2), float(y2), yaw_rad, gamma_rad)
                        if traj is None or traj.shape[0] <= 1:
                            continue
                        cond = np.array([x1, y1, float(x2), float(y2)], dtype=np.float32)
                        quadrant = self._get_quadrant(yaw_rad)

                        # store sample in current part buffer and record index pointing to this part
                        fname_rel = os.path.join("samples", current_part_name)
                        index_list.append({
                            "filename": fname_rel,
                            "quadrant": int(quadrant),
                            "length": int(traj.shape[0]),
                            "idx": idx,
                        })
                        archive_buf[f"traj_{idx}"] = traj
                        archive_buf[f"cond_{idx}"] = cond
                        archive_buf[f"yaw_{idx}"] = np.float32(yaw_rad)
                        archive_buf[f"gamma_{idx}"] = np.float32(gamma_rad)
                        archive_buf[f"quadrant_{idx}"] = np.int32(quadrant)
                        idx += 1

                        # flush buffer periodically to avoid large memory usage
                        if idx % FLUSH_EVERY == 0:
                            all_path = os.path.join(self.SAMPLES_DIR, current_part_name)
                            print(f"Flushing {len(archive_buf)/5} samples to archive: {all_path}")
                            np.savez_compressed(all_path, **archive_buf)
                            archive_buf = {}
                            part_idx += 1
                            current_part_name = f"all_samples_part{part_idx:03d}.npz"
                # optional progress print (per x slice)
            print(f"Progress: x loop finished; Total samples so far: {idx}")

        # Flush any remaining buffered samples into the final part file
        if len(archive_buf) > 0:
            all_path = os.path.join(self.SAMPLES_DIR, current_part_name)
            print(f"Flushing remaining {len(archive_buf)/5} samples to archive: {all_path}")
            np.savez_compressed(all_path, **archive_buf)
            archive_buf = {}
            
        # Save index (list of metadata dicts)
        np.save(self.INDEX_FILE, np.array(index_list, dtype=object))
        print(f"Data generation complete. {len(index_list)} samples indexed (split across {part_idx+1} archive files)")

    # -------------------------
    # Dataset API
    # -------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        fpath = os.path.join(self.DATA_ROOT, meta["filename"])
        # support two index formats:
        # 1) per-sample files (old): index entry points to a single-file containing keys 'traj','cond',...
        # 2) single archive (new): index entry has 'idx' and filename points to the archive; samples are stored
        #    under keys like 'traj_{idx}', 'cond_{idx}', 'yaw_{idx}', etc.
        with np.load(fpath, allow_pickle=False) as npz:
            if 'idx' in meta:
                k = int(meta['idx'])
                traj = npz[f"traj_{k}"].astype(np.float32)
                cond = npz[f"cond_{k}"].astype(np.float32)
                yaw = float(npz[f"yaw_{k}"].astype(np.float32))
                gamma = float(npz[f"gamma_{k}"].astype(np.float32))
                quadrant = int(npz[f"quadrant_{k}"].astype(np.int32))
            else:
                traj = npz["traj"].astype(np.float32)
                cond = npz["cond"].astype(np.float32)
                yaw = float(npz["yaw"].astype(np.float32))
                gamma = float(npz["gamma"].astype(np.float32))
                quadrant = int(npz["quadrant"].astype(np.int32))

        # return a small dict (collate_fn expects this)
        return {"traj": traj, "cond": cond, "yaw": yaw, "gamma": gamma, "quadrant": quadrant}

# -----------------------------
# Quadrant wrapper & loaders
# -----------------------------

class QuadrantDataset(Dataset):
    """
    Wraps DubinsDataset but exposes only samples from a given quadrant.
    Accepts either a list of global indices (preferred) or builds indices by scanning dataset index.
    """
    def __init__(self, base_dataset: DubinsDataset, quadrant_id: int, indices: list = None):
        self.base = base_dataset
        if indices is not None:
            self.indices = indices
        else:
            self.indices = [i for i, meta in enumerate(self.base.index) if int(meta['quadrant']) == int(quadrant_id)]
        self.quadrant = int(quadrant_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # map local idx -> global index
        global_idx = self.indices[idx]
        return self.base[global_idx]


def build_quadrant_loaders(dataset: DubinsDataset, batch_size: int, val_split: float = 0.2, shuffle_index=True):
    """
    Build DataLoaders for each quadrant. Splits indices into train/val at dataset-level, then
    builds QuadrantDataset wrappers for each quadrant and each split.
    Returns:
        train_loaders: list of 4 DataLoaders (one per quadrant)
        val_loaders:   same
        train_indices, val_indices: lists for reproducibility (optional)
    """
    # create list of indices per quadrant
    quad_indices = {0: [], 1: [], 2: [], 3: []}
    for i, meta in enumerate(dataset.index):
        q = int(meta["quadrant"])
        quad_indices[q].append(i)

    # remove samples with zero indices
    for q in quad_indices:
        quad_indices[q] = [i for i in quad_indices[q] if i != 0]
    
    # Limit max samples per quadrant for faster testing
    max_samples_per_quad = 10000
    for q in quad_indices:
        quad_indices[q] = quad_indices[q][:max_samples_per_quad]

    train_loaders = []
    val_loaders = []
    train_idx_by_quad = {}
    val_idx_by_quad = {}

    for q in range(4):
        idxs = quad_indices[q]
        if shuffle_index:
            random.shuffle(idxs)
        n_val = int(len(idxs) * val_split)
        val_idxs = idxs[:n_val]
        train_idxs = idxs[n_val:]
        train_idx_by_quad[q] = train_idxs
        val_idx_by_quad[q] = val_idxs

        train_ds_q = QuadrantDataset(dataset, quadrant_id=q, indices=train_idxs)
        val_ds_q = QuadrantDataset(dataset, quadrant_id=q, indices=val_idxs)

        train_loader_q = DataLoader(train_ds_q, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader_q = DataLoader(val_ds_q, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        train_loaders.append(train_loader_q)
        val_loaders.append(val_loader_q)

    return train_loaders, val_loaders, train_idx_by_quad, val_idx_by_quad

# -----------------------------
# Collate function (same API used earlier)
# -----------------------------
def collate_fn(batch):
    """
    Batch is a list of items; each item is dict with keys 'traj' (np.ndarray), 'cond' (np.ndarray), maybe others.
    Returns:
        padded: (B, L_max, feat) float tensor on device
        lengths: (B,) long tensor on CPU (we put lengths on device in training loop as needed)
        conds: (B, 4) float tensor on same device as padded
    """
    trajs = []
    conds = []
    for item in batch:
        # item['traj'] is np.ndarray
        trajs.append(torch.from_numpy(item['traj']))
        conds.append(torch.from_numpy(item['cond']))

    lengths = torch.tensor([t.size(0) for t in trajs], dtype=torch.long)
    batch_size = len(trajs)
    max_len = int(lengths.max().item()) if batch_size > 0 else 0
    feat = trajs[0].size(1) if trajs[0].dim() > 1 else 1

    padded = torch.zeros(batch_size, max_len, feat, dtype=torch.float32)
    for i, t in enumerate(trajs):
        L = t.size(0)
        padded[i, :L, :] = t.float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded = padded.to(device=device)
    conds = torch.stack(conds).float().to(device=device)

    return padded, lengths, conds

# -----------------------------
# Model (unchanged)
# -----------------------------
class DubinsLSTM(nn.Module):
    def __init__(self, input_dim=3, cond_dim=4, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_cond = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, conds, target_seq=None, lengths=None, teacher_forcing_ratio=0.5, seq_len: int = 50):
        B = conds.size(0)
        device = conds.device
        cond_embed = self.fc_cond(conds)
        cond_embed = cond_embed.unsqueeze(1)  # (B, 1, hidden_dim)
        if target_seq is not None:
            run_len = target_seq.size(1)
        else:
            run_len = seq_len

        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        out_seq = []

        prev_out = torch.zeros(B, 1, 3, device=device)
        for t in range(run_len):
            rnn_in = torch.cat([prev_out, cond_embed], dim=-1)
            out, (h, c) = self.lstm(rnn_in, (h, c))
            pred = self.fc_out(out)
            out_seq.append(pred)

            if target_seq is not None:
                if lengths is None:
                    use_tf = (random.random() < teacher_forcing_ratio)
                    prev_out = target_seq[:, t:t+1, :] if use_tf else pred
                else:
                    rand = torch.rand(B, device=device)
                    use_tf = (rand < teacher_forcing_ratio) & (t < lengths)
                    use_tf = use_tf.view(B, 1, 1)
                    prev_target = target_seq[:, t:t+1, :]
                    prev_out = torch.where(use_tf, prev_target, pred)
            else:
                prev_out = pred

        return torch.cat(out_seq, dim=1)

# -----------------------------
# Metrics & train/eval
# -----------------------------
def ADE(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def FDE(pred, gt):
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=-1))

def train_epoch(model, loader, optim_obj, device, criterion, teacher_forcing=0.5, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    for trajs, lengths, conds in loader:
        trajs = trajs.float().to(device)
        lengths = lengths.to(device)
        conds = conds.float().to(device)

        optim_obj.zero_grad()
        preds = model(conds, target_seq=trajs, lengths=lengths, teacher_forcing_ratio=teacher_forcing)

        if isinstance(criterion, nn.MSELoss) and criterion.reduction == 'none':
            loss_elem = criterion(preds, trajs)
        else:
            loss_elem = (preds - trajs) ** 2
        loss_t = loss_elem.mean(dim=-1)
        max_len = preds.size(1)
        mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        loss = (loss_t * mask).sum() / mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim_obj.step()
        running_loss += float(loss.item()) * trajs.size(0)
    return running_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0

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

            if isinstance(criterion, nn.MSELoss) and criterion.reduction == 'none':
                loss_elem = criterion(preds, trajs)
            else:
                loss_elem = (preds - trajs) ** 2
            loss_t = loss_elem.mean(dim=-1)
            max_len = preds.size(1)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
            loss = (loss_t * mask).sum() / mask.sum()
            running_loss += float(loss.item()) * trajs.size(0)

            norms = torch.norm(preds - trajs, dim=-1)
            per_seq_ADE = (norms * mask).sum(dim=1) / lengths.float()
            total_ADE += float(per_seq_ADE.sum().item())

            idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(preds.size(0), device=device)
            last_preds = preds[batch_idx, idx, :]
            last_gt = trajs[batch_idx, idx, :]
            fde_per = torch.norm(last_preds - last_gt, dim=-1)
            total_FDE += float(fde_per.sum().item())
    n = len(loader.dataset)
    return (running_loss / n if n > 0 else 0.0), (total_ADE / n if n > 0 else 0.0), (total_FDE / n if n > 0 else 0.0)

# -----------------------------
# Plotting helper
# -----------------------------
def plot_prediction_example(model, dataset, device=None, idx=0):
    model.eval()
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

# -----------------------------
# Main training loop
# -----------------------------
def main_train(batch_size=64, epochs=10, lr=1e-3, tf_ratio=0.5,
               early_stopping_patience: int = 3, early_stopping_min_delta: float = 1e-4,
               regenerate_dataset=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset (on-disk). Set regenerate_dataset=True to re-create samples.
    dataset = DubinsDataset(regenerate=regenerate_dataset)

    # Build quadrant loaders (train/val per quadrant)
    train_loaders, val_loaders, train_idxs, val_idxs = build_quadrant_loaders(dataset, batch_size=batch_size, val_split=0.2)

    model = DubinsLSTM().to(device)
    optim_obj = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    history = {'train_loss': [], 'val_loss': [], 'ADE': [], 'FDE': []}
    best_val = float('inf')
    epochs_no_improve = 0
    best_model_path = 'dubin_lstm_best.pth'

    # Training strategy: cycle through quadrants each epoch (you can change it)
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        # iterate quadrants sequentially for this epoch
        for q in range(4):
            loader_q = train_loaders[q]
            print(f"[Epoch {epoch+1}] Training on quadrant {q} (samples: {len(loader_q.dataset)})")
            if len(loader_q.dataset) == 0:
                print(f"  Skipping quadrant {q} (no training samples).")
                continue
            train_loss_q = train_epoch(model, loader_q, optim_obj, device, criterion, teacher_forcing=tf_ratio, clip_grad=2.0)
            epoch_train_loss += train_loss_q * len(loader_q.dataset)

        # compute average train loss across all train samples
        total_train_samples = sum(len(d.dataset) for d in train_loaders)
        avg_train_loss = (epoch_train_loss / total_train_samples) if total_train_samples > 0 else 0.0

        # Validation: evaluate across all quadrants (concatenate effect)
        val_loss_total = 0.0
        val_ADE_total = 0.0
        val_FDE_total = 0.0
        total_val_samples = 0
        for q in range(4):
            vloader = val_loaders[q]
            if len(vloader.dataset) == 0:
                continue
            v_loss, v_ADE, v_FDE = eval_epoch(model, vloader, device, criterion)
            n_v = len(vloader.dataset)
            val_loss_total += v_loss * n_v
            val_ADE_total += v_ADE * n_v
            val_FDE_total += v_FDE * n_v
            total_val_samples += n_v

        avg_val_loss = (val_loss_total / total_val_samples) if total_val_samples > 0 else 0.0
        avg_val_ADE = (val_ADE_total / total_val_samples) if total_val_samples > 0 else 0.0
        avg_val_FDE = (val_FDE_total / total_val_samples) if total_val_samples > 0 else 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['ADE'].append(avg_val_ADE)
        history['FDE'].append(avg_val_FDE)

        print(f"[{epoch+1:02d}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | ADE: {avg_val_ADE:.4f} | FDE: {avg_val_FDE:.4f}")

        if avg_val_loss < best_val - early_stopping_min_delta:
            best_val = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Val improved -> saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered (no improvement for {epochs_no_improve} epochs).")
            break

    # Load best model if exists
    if os.path.exists(best_model_path):
        print(f"Loading best model from '{best_model_path}'")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # save final model
    torch.save(model.state_dict(), "dubin_lstm.pth")
    print("Training complete. Model saved as 'dubin_lstm.pth'.")

    # plot training curves
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    return model, dataset, train_loaders, val_loaders, history

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # To force regeneration: main_train(..., regenerate_dataset=True)
    model, dataset, train_loaders, val_loaders, history = main_train(batch_size=4096, epochs=100, lr=1e-3, 
                                                                     tf_ratio=0.5, regenerate_dataset=False, 
                                                                     early_stopping_patience=3)
    # Example plot
    # pick a quadrant with at least one sample
    for q in range(4):
        if len(train_loaders[q].dataset) > 0:
            sample_global_idx = train_loaders[q].dataset.indices[0]
            plot_prediction_example(model, dataset, idx=sample_global_idx)
            

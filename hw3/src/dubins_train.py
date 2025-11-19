#!/usr/bin/env python3
"""
Full script with quadrant-by-yaw (Option B) data loader that saves samples to disk
and loads trajectories on demand to reduce RAM usage.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import os
import random
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchinfo import summary
from dubin_lstm_encoder_decoder import DubinsLSTMEncoderDecoder
from torch.utils.tensorboard import SummaryWriter

try:
    import matplotlib.pyplot as plt
except Exception:
    # Some test environments have NumPy/Matplotlib ABI mismatches. Allow importing the module
    # without plotting support so unit tests can run headless.
    plt = None
    print("Warning: matplotlib.pyplot not available - plotting functions will be disabled.")

try:
    from dubinEHF3d import dubinEHF3d
except Exception:
    # Try alternative import paths (tests may execute with different sys.path setups).
    try:
        from hw3.src.dubinEHF3d import dubinEHF3d
    except Exception:
        # Provide a safe stub so modules that import this file don't fail when they
        # only use dataset/statistics functionality (tests often don't run the path planner).
        def dubinEHF3d(*args, **kwargs):
            return None, None, 0

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
        [{'filename': 'samples/00000000.npz', 'quadrant':0, 'length':N, 'cond':[x1,y1,x2,y2,yaw,gamma]}, ...]

    Note: index entries may include a 'cond' field (list) containing [x1,y1,x2,y2,yaw,gamma].
    When present the dataset will prefer this metadata-stored condition over reading the
    archive entry for faster access.
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

    def __init__(self, regenerate: bool = False, r_min=100, step_length=10, data_root: str = None, 
                 normalize: bool = True, norm_eps: float = 1e-8, max_samples: int = 10000, samples_per_part: int = 500):
        """
        If regenerate=True, existing index/samples will be overwritten by new generation.
        """
        super().__init__()
        self.r_min = r_min
        self.step_length = step_length
        # allow overriding Data root (useful for tests)
        if data_root is not None:
            self.DATA_ROOT = data_root
            self.SAMPLES_DIR = os.path.join(self.DATA_ROOT, "samples")
            self.INDEX_FILE = os.path.join(self.DATA_ROOT, "index.npy")
        # normalization settings
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        self.NORM_FILE = os.path.join(self.DATA_ROOT, "norm_stats.npz")
        # generation controls (only used when regenerate=True)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.samples_per_part = int(samples_per_part)

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

        # # skip generation if index exists
        # if os.path.exists(self.INDEX_FILE):
        #     print("Index file already exists; skipping data generation.")
        #     return
        
        # Delete the index and samples if they exist
        if os.path.exists(self.INDEX_FILE):
            try:
                os.remove(self.INDEX_FILE)
            except Exception:
                pass

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
        FLUSH_EVERY = max(1, int(self.samples_per_part))  # how many samples per archive part
        current_part_name = f"all_samples_part{part_idx:03d}.npz"

        yaw_values = list(range(0, self.MAX_YAW, self.YAW_STEP))  # e.g., 0..350
        gamma_values = list(range(-self.MAX_MIN_ANGLE, self.MAX_MIN_ANGLE + 1, self.MAX_MIN_ANGLE_STEP))

        expected = ((2 * self.MAX_GRID) // self.X_Y_SPACE + 1) ** 2 * len(yaw_values) * len(gamma_values)
        print(f"Expected grid samples (upper bound): {expected}")
        debug = True
        debug_stopping_point = 100000
        stop = False
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
                        
                        cond = np.array([x1, y1, float(x2), float(y2), yaw_rad, gamma_rad], dtype=np.float32)
                        quadrant = self._get_quadrant(yaw_rad)

                        # store sample in current part buffer and record index pointing to this part
                        fname_rel = os.path.join("samples", current_part_name)
                        # store cond in the index so downstream code can access it without
                        # opening the archive. Convert to a Python list for safe storage.
                        index_list.append({
                            "filename": fname_rel,
                            "quadrant": int(quadrant),
                            "length": int(traj.shape[0]),
                            "idx": idx,
                            "cond": cond.tolist(),
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

        # Flush any remaining buffered samples into the final part file
        if len(archive_buf) > 0:
            all_path = os.path.join(self.SAMPLES_DIR, current_part_name)
            print(f"Flushing remaining {len(archive_buf)/5} samples to archive: {all_path}")
            np.savez_compressed(all_path, **archive_buf)
            archive_buf = {}
            
        # Save index (list of metadata dicts)
        np.save(self.INDEX_FILE, np.array(index_list, dtype=object))
        print(f"Data generation complete. {len(index_list)} samples indexed (split across {part_idx+1} archive files)")

    def denormalize_traj(self, traj_np: np.ndarray) -> np.ndarray:
        """Inverse transform for trajectory numpy arrays: (N,3) -> original scale."""
        if not getattr(self, 'normalize', False):
            return traj_np
        # traj was normalized as (traj - min_v) / denom where min_v = -MAX_GRID
        min_v = -float(self.MAX_GRID)
        max_v = float(self.MAX_GRID)
        denom = max_v - min_v if (max_v - min_v) != 0.0 else 1.0
        return (traj_np * denom) + min_v

    def denormalize_cond(self, cond_np: np.ndarray) -> np.ndarray:
        """Inverse transform for condition numpy arrays: (8,) or (B,8) -> original scale.
            Condition vector is [x1, y1, x2, y2, yaw, gamma, sin(yaw), cos(yaw)].
            This mirrors the cond_full used internally (cond + [sin(yaw), cos(yaw)]).
        """
        if not getattr(self, 'normalize', False):
            return cond_np
        # handle batch or single
        arr = np.asarray(cond_np, dtype=np.float32)
        min_v = -float(self.MAX_GRID)
        max_v = float(self.MAX_GRID)
        denom = max_v - min_v if (max_v - min_v) != 0.0 else 1.0

        def denorm_single(a):
            out = a.copy()
            # positions
            out[:4] = (out[:4] * denom) + min_v
            # yaw, gamma
            out[4] = out[4] * (2.0 * np.pi)
            out[5] = out[5] * (2.0 * np.pi)
            # sin/cos: if we have stats, invert standardization; otherwise leave as-is
            # if hasattr(self, 'cond_mean') and hasattr(self, 'cond_std'):
            #     out[6:] = (out[6:] * self.cond_std[6:]) + self.cond_mean[6:]
            return out

        if arr.ndim == 1:
            return denorm_single(arr)
        else:
            return np.stack([denorm_single(row) for row in arr], axis=0)

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
                # Prefer condition stored in the index to avoid reopening archive entries
                if 'cond' in meta:
                    cond = np.asarray(meta['cond'], dtype=np.float32)
                    # cond layout when stored in index: [x1,y1,x2,y2,yaw,gamma]
                    yaw = float(cond[4])
                    gamma = float(cond[5])
                else:
                    cond = npz[f"cond_{k}"].astype(np.float32)
                    yaw = float(npz[f"yaw_{k}"].astype(np.float32))
                    gamma = float(npz[f"gamma_{k}"].astype(np.float32))
                quadrant = int(npz[f"quadrant_{k}"].astype(np.int32))
            else:
                traj = npz["traj"].astype(np.float32)
                if 'cond' in meta:
                    cond = np.asarray(meta['cond'], dtype=np.float32)
                    yaw = float(cond[4])
                    gamma = float(cond[5])
                else:
                    cond = npz["cond"].astype(np.float32)
                    yaw = float(npz["yaw"].astype(np.float32))
                    gamma = float(npz["gamma"].astype(np.float32))
                quadrant = int(npz["quadrant"].astype(np.int32))
        # build full cond vector including sin(yaw), cos(yaw) and gamma
        sy = float(np.sin(yaw))
        cy = float(np.cos(yaw))
        cond_full = np.concatenate([cond, np.array([sy, cy], dtype=np.float32)])

        # apply normalization if requested
        if getattr(self, 'normalize', False):
            # traj: (N,3), cond_full: (7,)
            # Normalize traj using dataset bounds: min = -MAX_GRID, max = +MAX_GRID
            # formula: (value - min) / (max - min)
            min_v = -float(self.MAX_GRID)
            max_v = float(self.MAX_GRID)
            denom = max_v - min_v if (max_v - min_v) != 0.0 else 1.0
            traj = (traj - min_v) / denom

            # Normalize position entries in cond_full (x1,y1,x2,y2) using same MIN/MAX mapping
            # cond_full layout: [x1, y1, x2, y2, yaw, gamma, sin(yaw), cos(yaw)]
            cond_full = cond_full.astype(np.float32)
            # normalize x/y coordinates to [0,1]
            cond_full[:4] = (cond_full[:4] - min_v) / denom
            # normalize yaw by wrapping negative radians into [0,2*pi) then mapping to [0,1)
            # gamma remains as radians mapped by 2*pi (can be negative)
            # cond_full layout: [x1, y1, x2, y2, yaw, gamma, sin(yaw), cos(yaw)]
            yaw_wrapped = (float(yaw) + 2.0 * np.pi) % (2.0 * np.pi)
            cond_full[4] = yaw_wrapped / (2.0 * np.pi)
            gamma_wrapped = (float(gamma) + 2.0 * np.pi) % (2.0 * np.pi)
            cond_full[5] = gamma_wrapped / (2.0 * np.pi)

        # return a small dict (collate_fn expects this)
        return {"traj": traj, "cond": cond_full, "yaw": yaw, "gamma": gamma, "quadrant": quadrant}

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

class BucketBatchSampler(torch.utils.data.Sampler):
    """Bucket-based batch sampler that groups similar-length samples to reduce padding.

    indices: list of global indices (or local indices within a QuadrantDataset) to sample from
    lengths_map: mapping from global index -> sequence length
    bucket_size: how many indices to group into a bucket before batching
    """
    def __init__(self, indices, lengths_map, batch_size, bucket_size=100, shuffle=True):
        self.indices = list(indices)
        self.lengths_map = lengths_map
        self.batch_size = batch_size
        self.bucket_size = max(1, int(bucket_size))
        self.shuffle = shuffle

    def __iter__(self):
        inds = list(self.indices)
        # sort by length to reduce padding inside buckets
        inds.sort(key=lambda i: int(self.lengths_map.get(i, 0)))
        # split into buckets
        buckets = [inds[i:i + self.bucket_size] for i in range(0, len(inds), self.bucket_size)]
        if self.shuffle:
            random.shuffle(buckets)
        for bucket in buckets:
            # optionally shuffle within bucket to add randomness
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        if len(self.indices) == 0:
            return 0
        # approximate number of batches
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b 
                   in [self.indices[i:i + self.bucket_size] for i in range(0, len(self.indices), self.bucket_size)])

def build_quadrant_loaders(dataset: DubinsDataset, batch_size: int, val_split: float = 0.15, test_split: float = 0.15, shuffle_index=True, dynamic_batching=False, bucket_size=100):
    """
    Build DataLoaders for each quadrant. Splits indices into train/val at dataset-level, then
    builds QuadrantDataset wrappers for each quadrant and each split.
    Returns:
        train_loaders: list of 4 DataLoaders (one per quadrant)
        val_loaders:   list of 4 DataLoaders
        test_loaders:  list of 4 DataLoaders (may be empty if test_split==0)
        train_indices, val_indices, test_indices: lists for reproducibility (optional)
    """
    # create list of indices per quadrant
    quad_indices = {0: [], 1: [], 2: [], 3: []}
    for i, meta in enumerate(dataset.index):
        q = int(meta["quadrant"])
        quad_indices[q].append(i)

    # (previously removed index 0 samples here; keep all indices intact)
    
    # Limit max samples per quadrant for faster testing
    max_samples_per_quad = 10000
    for q in quad_indices:
        # quad_indices[q] = quad_indices[q][:max_samples_per_quad]
        quad_indices[q] = quad_indices[q][:]

    train_loaders = []
    val_loaders = []
    test_loaders = []
    train_idx_by_quad = {}
    val_idx_by_quad = {}
    test_idx_by_quad = {}

    for q in range(4):
        idxs = quad_indices[q]
        if shuffle_index:
            random.shuffle(idxs)
        n_val = int(len(idxs) * val_split)
        n_test = int(len(idxs) * test_split)
        # allocate: first val, then test, rest train (deterministic after shuffle)
        val_idxs = idxs[:n_val]
        test_idxs = idxs[n_val:n_val + n_test]
        train_idxs = idxs[n_val + n_test:]

        train_idx_by_quad[q] = train_idxs
        val_idx_by_quad[q] = val_idxs
        test_idx_by_quad[q] = test_idxs

        train_ds_q = QuadrantDataset(dataset, quadrant_id=q, indices=train_idxs)
        val_ds_q = QuadrantDataset(dataset, quadrant_id=q, indices=val_idxs)
        test_ds_q = QuadrantDataset(dataset, quadrant_id=q, indices=test_idxs)
        if dynamic_batching:
            # Build lengths map for global indices so sampler can sort by true sequence lengths
            # Build lengths map for local indices within this quadrant's index list
            # BucketBatchSampler expects indices relative to the dataset it's sampling from
            train_local_indices = list(range(len(train_idxs)))
            val_local_indices = list(range(len(val_idxs)))
            test_local_indices = list(range(len(test_idxs)))
            train_lengths_map = {li: int(dataset.index[train_idxs[li]]['length']) for li in train_local_indices}
            val_lengths_map = {li: int(dataset.index[val_idxs[li]]['length']) for li in val_local_indices}
            test_lengths_map = {li: int(dataset.index[test_idxs[li]]['length']) for li in test_local_indices}

            train_sampler = BucketBatchSampler(train_local_indices, train_lengths_map, batch_size, bucket_size=bucket_size, shuffle=True)
            val_sampler = BucketBatchSampler(val_local_indices, val_lengths_map, batch_size, bucket_size=bucket_size, shuffle=False)
            test_sampler = BucketBatchSampler(test_local_indices, test_lengths_map, batch_size, bucket_size=bucket_size, shuffle=False)

            train_loader_q = DataLoader(train_ds_q, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
            val_loader_q = DataLoader(val_ds_q, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=4)
            test_loader_q = DataLoader(test_ds_q, batch_sampler=test_sampler, collate_fn=collate_fn, num_workers=4)
        else:
            train_loader_q = DataLoader(train_ds_q, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
            val_loader_q = DataLoader(val_ds_q, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
            test_loader_q = DataLoader(test_ds_q, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

        train_loaders.append(train_loader_q)
        val_loaders.append(val_loader_q)
        test_loaders.append(test_loader_q)

    return train_loaders, val_loaders, test_loaders, train_idx_by_quad, val_idx_by_quad, test_idx_by_quad

# -----------------------------
# Collate function (same API used earlier)
# -----------------------------
def collate_fn(batch):
    """
    Batch is a list of items; each item is dict with keys 'traj' (np.ndarray), 'cond' (np.ndarray), maybe others.
    Returns:
        padded: (B, L_max, feat) float tensor on device
        lengths: (B,) long tensor on CPU (we put lengths on device in training loop as needed)
    conds: (B, 8) float tensor on same device as padded (condition: [x1,y1,x2,y2,yaw,gamma,sin(yaw),cos(yaw)])
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

    # Keep tensors on CPU here; caller (training/eval) should move to the desired device.
    conds = torch.stack(conds).float()

    return padded, lengths, conds

# -----------------------------
# Model (unchanged)
# -----------------------------
class DubinsLSTM(nn.Module):
    def __init__(self, input_dim=3, cond_dim=8, hidden_dim=64, num_layers=6, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_cond = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, conds, target_seq=None, lengths=None, teacher_forcing_ratio=0.5, 
                seq_len: int = 50):
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

def train_epoch(model, loader, optim_obj, device, criterion, teacher_forcing=0.0, clip_grad=1.0):
    """
    Train for one epoch over the provided loader.
    Returns: avg_loss
    """
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

    avg = running_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    return avg

def eval_epoch(model, loader, device, criterion):
    """
    Evaluate model on loader. Returns: (avg_loss, avg_ADE, avg_FDE)
    """
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
    avg_loss = (running_loss / n if n > 0 else 0.0)
    avg_ADE = (total_ADE / n if n > 0 else 0.0)
    avg_FDE = (total_FDE / n if n > 0 else 0.0)
    return avg_loss, avg_ADE, avg_FDE

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
    # If dataset returns normalized values, denormalize for plotting
    try:
        if getattr(dataset, 'normalize', False):
            preds = dataset.denormalize_traj(preds)
            gt = dataset.denormalize_traj(gt)
    except Exception:
        # be robust to datasets that don't implement denormalize helpers
        pass
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
def main(batch_size=64, epochs=10, lr=1e-3, tf_ratio=0.5,
         early_stopping_patience: int = 3, early_stopping_min_delta: float = 1e-4,
         regenerate_dataset=False,
         load_model_path: str = None,
         evaluate_only: bool = False,
         inference_only: bool = False,
         dynamic_batching: bool = False,
         bucket_size: int = 100,
         data_root: str = None,
         model_dir: str = None,
         model_version: int = 1,
         hist_interval: int = 2,
         histograms_enabled: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset (on-disk). Set regenerate_dataset=True to re-create samples.
    dataset = DubinsDataset(regenerate=regenerate_dataset, data_root=data_root) if data_root is not None else DubinsDataset(regenerate=regenerate_dataset)

    # Build quadrant loaders (train/val/test per quadrant)
    train_loaders, val_loaders, test_loaders, train_idxs, val_idxs, test_idxs = build_quadrant_loaders(dataset, batch_size=batch_size, val_split=0.2, dynamic_batching=dynamic_batching, bucket_size=bucket_size)

    if model_version == 1:
        model = DubinsLSTM().to(device)
    else:
        model = DubinsLSTMEncoderDecoder().to(device)

    # summary(model, input_size=(batch_size, 8))
    optim_obj = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    # Prepare TensorBoard writer (logs go under model_dir or cwd)
    model_dir_to_use = model_dir if model_dir is not None else os.getcwd()
    os.makedirs(model_dir_to_use, exist_ok=True)
    try:
        log_subdir = os.path.join(model_dir_to_use, 'runs', time.strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_subdir)
    except Exception:
        writer = None
    # effective histogram interval; set to 0 to disable
    effective_hist_interval = int(hist_interval) if histograms_enabled else 0

    # Optionally load existing model weights (for evaluation or warm-start)
    # If user did not pass --load-model-path, try to auto-detect a local 'dubin_lstm.pth'
    if load_model_path is None:
        candidates = []
        if model_dir is not None:
            candidates.append(os.path.join(model_dir, 'dubin_lstm.pth'))
        candidates.append(os.path.join(os.getcwd(), 'dubin_lstm.pth'))
        for cand in candidates:
            if os.path.exists(cand):
                load_model_path = cand
                print(f"Auto-detected model file at '{cand}' (set as load_model_path)")
                break

    if load_model_path is not None and os.path.exists(load_model_path):
        print(f"Loading model weights from '{load_model_path}'")
        model.load_state_dict(torch.load(load_model_path, map_location=device, weights_only=True))
        if inference_only:
            print("Inference-only mode: skipping training and evaluation.")
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            return model, dataset, train_loaders, val_loaders, test_loaders, {}

        # If user only wanted to evaluate, skip training and run evaluation on val loaders
        if evaluate_only:
            print("Evaluate-only mode: running evaluation on validation and test loaders...")
            # aggregate evaluation across quadrants (validation)
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
            print(f"Validation -- Loss: {avg_val_loss:.6f} | ADE: {avg_val_ADE:.4f} | FDE: {avg_val_FDE:.4f}")

            # aggregate evaluation across quadrants (test)
            test_loss_total = 0.0
            test_ADE_total = 0.0
            test_FDE_total = 0.0
            total_test_samples = 0
            for q in range(4):
                tloader = test_loaders[q]
                if len(tloader.dataset) == 0:
                    continue
                t_loss, t_ADE, t_FDE = eval_epoch(model, tloader, device, criterion)
                n_t = len(tloader.dataset)
                test_loss_total += t_loss * n_t
                test_ADE_total += t_ADE * n_t
                test_FDE_total += t_FDE * n_t
                total_test_samples += n_t

            avg_test_loss = (test_loss_total / total_test_samples) if total_test_samples > 0 else 0.0
            avg_test_ADE = (test_ADE_total / total_test_samples) if total_test_samples > 0 else 0.0
            avg_test_FDE = (test_FDE_total / total_test_samples) if total_test_samples > 0 else 0.0
            print(f"Test       -- Loss: {avg_test_loss:.6f} | ADE: {avg_test_ADE:.4f} | FDE: {avg_test_FDE:.4f}")
            # return with empty history for evaluate-only
            return model, dataset, train_loaders, val_loaders, test_loaders, {}

    history = {'train_loss': [], 'val_loss': [], 'ADE': [], 'FDE': []}
    best_val = float('inf')
    epochs_no_improve = 0
    # model_dir override (use temp dir in tests)
    # model_dir_to_use was created above when initializing the writer
    best_model_path = os.path.join(model_dir_to_use, 'dubin_lstm_best.pth')

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

        # TensorBoard: log epoch-level scalars
        if writer is not None:
            try:
                writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
                writer.add_scalar('val/ADE', avg_val_ADE, epoch)
                writer.add_scalar('val/FDE', avg_val_FDE, epoch)
                writer.flush()
                # Log parameter and gradient histograms (every `hist_interval` epochs)
                try:
                    if (hist_interval is not None) and (hist_interval > 0) and (epoch % hist_interval == 0):
                        for name, param in model.named_parameters():
                            # replace dots with slashes for nicer grouping in TB UI
                            tag = name.replace('.', '/')
                            try:
                                writer.add_histogram(f'params/{tag}', param.data.cpu().numpy(), epoch)
                            except Exception:
                                # fallback to tensor input if numpy conversion fails
                                try:
                                    writer.add_histogram(f'params/{tag}', param.data.cpu(), epoch)
                                except Exception:
                                    pass
                            if param.grad is not None:
                                try:
                                    writer.add_histogram(f'grads/{tag}', param.grad.data.cpu().numpy(), epoch)
                                except Exception:
                                    try:
                                        writer.add_histogram(f'grads/{tag}', param.grad.data.cpu(), epoch)
                                    except Exception:
                                        pass
                except Exception:
                    # avoid crashing training if histogram logging fails
                    pass
            except Exception:
                pass

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
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # save final model
    final_model_path = os.path.join(model_dir_to_use, 'dubin_lstm.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Model saved as '{final_model_path}'.")

    # plot training curves (only if matplotlib available and provides the plotting API)
    if plt is not None and hasattr(plt, 'figure'):
        try:
            plt.figure()
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title("Loss Curve")
            plt.show()
        except Exception:
            # plotting is optional; ignore any plotting errors in headless/test environments
            pass

    # Close TensorBoard writer if present
    if 'writer' in locals() and writer is not None:
        try:
            writer.close()
        except Exception:
            pass

    return model, dataset, train_loaders, val_loaders, test_loaders, history

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate Dubins LSTM model")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tf-ratio', type=float, default=0.5)
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--dynamic-batching', action='store_true')
    parser.add_argument('--bucket-size', type=int, default=100)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--load-model-path', type=str, default=None)
    parser.add_argument('--data-root', type=str, default=None, help='Override data root directory')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory to save model artifacts')
    parser.add_argument('--hist-interval', type=int, default=2, help='Epoch interval for histogram logging (0 to disable)')
    parser.add_argument('--no-histograms', action='store_true', help='Disable histogram logging')

    args = parser.parse_args()
    args.regenerate=False
    
    model, dataset, train_loaders, val_loaders, test_loaders, history = main(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        tf_ratio=args.tf_ratio,
        regenerate_dataset=args.regenerate,
        early_stopping_patience=args.early_stopping_patience,
        evaluate_only=args.evaluate_only,
        inference_only=False,
        load_model_path=args.load_model_path,
        dynamic_batching=args.dynamic_batching,
        bucket_size=args.bucket_size,
        data_root=args.data_root,
        model_dir=args.model_dir,
        hist_interval=args.hist_interval,
        histograms_enabled=(not args.no_histograms),
        model_version=2
    )

    # Example plot (only if matplotlib available)
    if plt is not None:
        # pick a quadrant with at least one sample
        for q in range(4):
            if len(train_loaders[q].dataset) > 0:
                # pick 10 samples to plot
                for x in range(2):
                    sample_global_idx = train_loaders[q].dataset.indices[x]
                    plot_prediction_example(model, dataset, idx=sample_global_idx)
            

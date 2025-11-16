# hw3 — Dubins LSTM training utilities

This folder (hw3) contains code and tests for a Dubins-like trajectory prediction LSTM pipeline. The README documents the features added to `hw3/src/dubins_train.py` and related tests, plus usage notes and troubleshooting tips.

## Key features implemented

- On-disk, indexed dataset format
  - Samples are stored in compressed NumPy archives under `data/samples/` (archive parts named like `all_samples_part000.npz`).
  - An index file `data/index.npy` (array of small metadata dicts) points to archive files and optional sample keys (e.g., `{ 'filename': 'samples/all_samples_part000.npz', 'idx': 0, 'length': 12, 'quadrant': 0 }`).
  - `DubinsDataset` loads samples on demand from these archives to reduce RAM usage.

- Collation for variable-length trajectories
  - `collate_fn(batch)` pads trajectories to the batch max length and returns:
    - padded trajectories: Tensor (B, L_max, 3)
    - lengths: LongTensor (B,)
    - conds: Tensor (B, 6)  — condition vector is now [x1, y1, x2, y2, yaw, gamma]
  - The collate function keeps tensors on CPU; training/eval routines move tensors to the selected device.

- Condition vector extended (yaw & gamma)
  - Condition arrays now include yaw and gamma in addition to [x1, y1, x2, y2]. The condition vector is 6-dimensional.
  - `DubinsLSTM` default `cond_dim` has been updated to 6 so the model accepts the expanded condition vector.

- Normalization support
  - `DubinsDataset(regenerate=False, data_root=..., normalize=True)` computes (or loads) normalization statistics and returns normalized trajectories and condition vectors.
  - Normalization statistics are saved to `data/norm_stats.npz` with keys: `traj_mean`, `traj_std`, `cond_mean`, `cond_std`.
  - Trajectory statistics are computed over all points from all trajectories (per-dimension mean/std for 3 dims). Condition statistics are computed per-sample over the 6-dim cond vector.
  - Zero-variance values (e.g., constant yaw/gamma) are clipped to `norm_eps` to avoid division-by-zero.
  - Helpers: `denormalize_traj()` and `denormalize_cond()` are provided to invert normalization for plotting or evaluation.

- Quadrant-aware splits and loaders
  - Samples are assigned to quadrants (0..3) based on yaw; the dataset supports a `quadrant` metadata field.
  - `QuadrantDataset` wraps `DubinsDataset` and exposes only the indices for a specific quadrant.
  - `build_quadrant_loaders()` builds per-quadrant train/val/test DataLoaders and returns them along with the split indices.

- Bucketed / dynamic batching
  - `BucketBatchSampler` groups similar-length samples into buckets and yields batches with less padding.
  - `build_quadrant_loaders(..., dynamic_batching=True, bucket_size=...)` will build DataLoaders using `BucketBatchSampler` for each split.

- Model and training
  - `DubinsLSTM` takes condition vectors (B,6) and produces trajectory sequences. The internal `fc_cond` projects condition vectors to an embedding that's concatenated to the LSTM input at each step.
  - Training/evaluation helpers `train_epoch()` and `eval_epoch()` implement masked MSE loss (ignoring padded timesteps) and compute ADE/FDE metrics.
  - Early stopping and saving: `main()` includes early stopping logic and saves best/final model weights to the provided `model_dir` (or CWD by default).

- CLI / main
  - `hw3/src/dubins_train.py` includes a `main()` function with CLI glue at the bottom. Options include dataset regeneration, dynamic batching, evaluate-only mode, bucket size, model directory, and training hyperparameters.

- Robust imports & plotting
  - Matplotlib imports are guarded so headless or ABI-incompatible environments can import the module without crashing; plotting functions will be disabled if matplotlib is not available.
  - The import of the path planner (`dubinEHF3d`) is made robust and falls back to a no-op stub if unavailable. This keeps unit tests and normalization logic importable in environments without the planner.

- Tests
  - Unit tests live under `hw3/src/` (e.g., `test_collate.py`, `test_train_step.py`, `test_loader_splits.py`, `test_dynamic_batching.py`, `test_main_return.py`, `test_normalization.py`).
  - Tests were updated to use the new 6-dim condition vectors where they construct in-memory batches.
  - `test_normalization.py` creates a tiny on-disk archive and index inside a pytest `tmp_path` and asserts normalization statistics and denormalization behavior (yaw/gamma zero-variance -> std clipped to `norm_eps`).

## Usage examples

- Run `main()` programmatically or via CLI (from project root):

```bash
python3 hw3/src/dubins_train.py --data-root ./data --model-dir ./models --epochs 10 --batch-size 64
```

- Create a dataset on disk
  - The repository expects either a pre-built `data/index.npy` and archived samples in `data/samples/`, or you can implement generation by setting `regenerate=True` and providing generation logic. Note: in this workspace the `_generate_samples()` method currently raises a clear RuntimeError (the generation routine was intentionally left out/disabled). If you need dataset generation, I can re-implement it.

- Load dataset with normalization

```python
from hw3.src.dubins_train import DubinsDataset
# compute (and cache) normalization stats
ds = DubinsDataset(regenerate=False, data_root='data', normalize=True)
item = ds[0]
traj_norm = item['traj']   # (N,3) normalized
cond_norm = item['cond']   # (6,) normalized
# To denormalize:
traj_orig = ds.denormalize_traj(traj_norm)
cond_orig = ds.denormalize_cond(cond_norm)
```

- Collate API (for custom DataLoader use)
  - `collate_fn(batch)` expects list of items where `item['traj']` is an (N,3) np.ndarray and `item['cond']` is (6,) np.ndarray. Returns (padded_trajs, lengths, conds).

- Model
  - Instantiate model: `model = DubinsLSTM()` (defaults: cond_dim=6). Forward signature:

```python
preds = model(conds, target_seq=trajs, lengths=lengths, teacher_forcing_ratio=0.5)
# or for generation:
preds = model(conds, target_seq=None, seq_len=50, teacher_forcing_ratio=0.0)
```

## Troubleshooting / notes

- NumPy / Matplotlib ABI: some environments (CI or system image) may have NumPy/Matplotlib ABI incompatibilities (NumPy 2.x vs extensions compiled for 1.x). The code guards matplotlib import and will disable plotting if import fails. If you plan to run plotting locally, ensure your environment has compatible NumPy/Matplotlib versions.

- `dubinEHF3d` import: the module attempts to import `dubinEHF3d` from the local `hw3/src` package or from top-level import; if not found a stub is used. If you rely on path planning to generate trajectories, make sure `dubinEHF3d.py` is importable and functional (and avoid importing heavy plotting modules if NumPy ABI conflicts exist).

- `_generate_samples()` is intentionally left as a stub that raises a runtime error in this runtime; providing a dataset index and archives is the recommended path for tests and quick iterations. If you want dataset regeneration enabled, ask and I will re-implement a safe generator that writes archives and index files.

## Suggested next steps

- Re-run the test suite in your environment:

```bash
# from repo root
pip install -r requirements.txt  # ensure pytest is installed if not present
pytest -q hw3/src
```

- If you want yaw to be represented differently (e.g., sin(yaw), cos(yaw) instead of raw angle), I can update both the dataset normalization and model to produce/use that representation.

- If you want `_generate_samples()` re-enabled, I can implement a deterministic generator that writes archives and index files in the correct format.

---

If you'd like, I can now run the test suite here (I can try to run pytest or run tests directly). Note: this environment may not have pytest installed and may show NumPy/Matplotlib ABI warnings; I can still run the test functions directly as I did in my validation run and report results.

If you want a different README location or additional docs (e.g., a short `HOWTO_generate_dataset.md`), tell me where to put it and I will add it next.

## Index format & condition ordering (note)

- Index entries may now include a `cond` field with 6 values: `[x1, y1, x2, y2, yaw, gamma]` stored as a JSON-style list in `data/index.npy` metadata entries.
- Internally the dataset builds a full condition vector `cond_full` by appending the trigonometric embedding: `[x1, y1, x2, y2, yaw, gamma, sin(yaw), cos(yaw)]` (8 values).  
- `denormalize_cond()` and the normalization stats operate on this 8-dimensional `cond_full` shape. The method accepts either a single array `(8,)` or a batch `(B,8)` and returns the denormalized values accordingly.
- When an index entry contains `meta['cond']`, `DubinsDataset` prefers that metadata and avoids re-reading condition arrays from the archive for faster access.

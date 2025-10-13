# HW2: ResNet152 Image Classification

## Features
- Train and fine-tune ResNet152 on your dataset
- Early stopping and best model saving
- TensorBoard logging for loss and accuracy
- Inference on raw test images (no ground truth required)
- Individual prediction images and a grid summary with prediction probabilities

## Methodology
- Images are resized to 224x224 and normalized using ImageNet statistics via `torchvision.transforms.Compose`.
- Training and validation sets use `ImageFolder`, which maps subfolder names to class indices.
- Test images are loaded as raw files using a custom generator, supporting flexible evaluation without ground truth labels.
- Two training modes: `fine_tune_last_layer()` (freezes all but the last layer for transfer learning) and `train_entire_model()` (unfreezes all layers for end-to-end retraining). Both use Adam optimizer at lr=0.001 and call `train_model()`.
- The training loop performs batch-wise forward/backward passes, accumulates loss/accuracy, logs metrics to TensorBoard, tracks validation performance, implements early stopping, and saves the best model checkpoint.

## Usage

### Install requirements
```
pip install -r requirements.txt
```

### Run options

**Fine-tune last layer:**
```
python src/hw2_main.py --mode finetune --epochs 5 --patience 3 --save_path best_finetune_model.pth
```

**Train entire model:**
```
python src/hw2_main.py --mode train --epochs 5 --patience 3 --save_path best_train_model.pth
```

**Test and save predictions:**
```
python src/hw2_main.py --mode test --num_images 10 --save_path best_finetune_model.pth
```

Arguments:
- `--mode`: Select 'finetune', 'train', or 'test'.
- `--epochs`: Number of epochs for training (default: 5).
- `--patience`: Early stopping patience (default: 3).
- `--save_path`: Path to save/load best model weights.
- `--num_images`: Number of test images to evaluate and save plots for (default: 10).

### Output
- Individual prediction images: `test_predictions/`
- Grid summary of predictions: `docs/test_predictions_grid_current.png` (current weights), `docs/test_predictions_grid_best.png` (best weights)
- Each prediction includes the probability/confidence

### TensorBoard
To visualize training progress:
```
tensorboard --logdir runs
```

### Notes
- Place your training, validation, and test images in:
  - `hw2/src/data/train/`
  - `hw2/src/data/val/`
  - `hw2/src/data/test/`
- Test images do not require ground truth labels or subfolders.
- The `.gitignore` excludes image outputs and prediction folders.

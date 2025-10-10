# Running hw2_main.py

You can run the main script with the following options:

**Fine-tune last layer:**
```
python src/hw2_main.py --mode finetune --epochs 5
```

**Train entire model:**
```
python src/hw2_main.py --mode train --epochs 5
```

**Test and save predictions:**
```
python src/hw2_main.py --mode test --num_images 10
```

Arguments:
- `--mode`: Select 'finetune', 'train', or 'test'.
- `--epochs`: Number of epochs for training (default: 5).
- `--num_images`: Number of test images to evaluate and save plots for (default: all).

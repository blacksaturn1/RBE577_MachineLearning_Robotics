HW2 Report: ResNet152 Image Classification
===========================================

Author: Jason Patel, Camilo Girgado
Date: October 2025

---

## Methodology

This project implements an image classification pipeline using a deep convolutional neural network, ResNet152, pretrained on ImageNet. The workflow includes:

- **Data Preparation:**
  - Training, validation, and test images are organized in separate folders.
  - Images are resized to 224x224 and normalized using ImageNet statistics.
  - Test images are loaded as raw files, allowing inference without ground truth labels.

- **Model Architecture:**
  - ResNet152 is used with its final fully connected layer replaced by a new layer for 10 output classes, followed by a softmax activation.
  - The model is transferred to GPU if available.

- **Training Procedure:**
  - Two modes: fine-tuning only the last layer, or training the entire model.
  - Adam optimizer is used for both modes.
  - Cross-entropy loss is used for classification.
  - Early stopping is implemented with a patience parameter to prevent overfitting.
  - The best model (lowest validation loss) is saved during training.
  - Training and validation loss/accuracy are logged to TensorBoard.

- **Inference and Output:**
  - The model can be run on test images with or without loading the best saved weights.
  - Individual prediction images and a grid summary are generated, showing predicted class and probability for each test image.

---

## Hyperparameters

- **Model:** ResNet152 (pretrained)
- **Image Size:** 224x224
- **Batch Size:** 32
- **Optimizer:** Adam
  - Fine-tune last layer: lr=0.001
  - Train entire model: lr=0.001
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 5 (default, configurable)
- **Early Stopping Patience:** 3 (default, configurable)
- **Save Path:** best_finetune_model.pth or best_train_model.pth (configurable)
- **Number of Test Images:** 10 (default, configurable)

---

## Lessons Learned

- **Transfer Learning:** Using a pretrained model like ResNet152 significantly accelerates convergence and improves accuracy, especially with limited data.
- **Early Stopping:** Implementing early stopping is crucial to avoid overfitting and unnecessary computation.
- **Data Normalization:** Proper normalization is essential for leveraging pretrained weights.
- **Flexible Inference:** Supporting raw test images without ground truth labels makes the pipeline adaptable for real-world deployment.
- **Visualization:** Generating both individual and grid prediction images provides valuable insights into model performance and failure cases.
- **Hyperparameter Tuning:** Exposing key hyperparameters as command-line options makes experimentation and reproducibility easier.
- **Automation:** Integrating TensorBoard and automated model saving streamlines the training and evaluation process.

---

## Results

- The best model and current model predictions are visualized in grid images (`test_predictions_grid_best.png` and `test_predictions_grid_current.png`).
- Each prediction includes the class label and probability, aiding in confidence assessment.
- Training and validation metrics are available in TensorBoard for further analysis.

---

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- TensorBoard Documentation: https://www.tensorflow.org/tensorboard


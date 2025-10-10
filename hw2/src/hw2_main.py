import os
import torch
from torchvision import models, transforms, datasets

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the largest ResNet model (ResNet-152) pretrained on ImageNet
resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 10),
    nn.Softmax(dim=1)
)
resnet = resnet.to(device)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

 # Load train and validation datasets (update paths as needed)
train_dataset = datasets.ImageFolder(root='./hw2/src/data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./hw2/src/data/val', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# Remove test_dataset and use raw image loader
from PIL import Image
import glob
test_image_paths = sorted(glob.glob('./hw2/src/data/test/*.*'))
def test_image_loader(image_paths, transform, batch_size=32):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [transform(Image.open(p).convert('RGB')) for p in batch_paths]
        yield torch.stack(images)
test_loader = test_image_loader(test_image_paths, transform, batch_size=32)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, log_dir='runs/exp1', patience=3, save_path='best_model.pth'):
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        avg_train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / total
        val_acc = correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Early stopping and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}')
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epoch(s)')
            if epochs_no_improve >= patience:
                print('Early stopping triggered.')
                break
    writer.close()

 # 1. Fine-tuning: Freeze all layers except the last layer
def fine_tune_last_layer(model, train_loader, val_loader, criterion, num_epochs=5, patience=3, save_path='best_model.pth'):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=patience, save_path=save_path)

 # 2. Train entire model
def train_entire_model(model, train_loader, val_loader, criterion, num_epochs=5, patience=3, save_path='best_model.pth'):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=patience, save_path=save_path)


# Test set evaluation
def evaluate_on_test(model, test_loader, num_images=None, class_names=None):
    model.eval()
    processed = 0
    with torch.no_grad():
        for batch in test_loader:
            # If test_loader returns (images, labels), ignore labels
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images = batch[0]
            else:
                images = batch
            if num_images is not None and processed >= num_images:
                break
            batch_size = images.size(0)
            if num_images is not None and processed + batch_size > num_images:
                batch_size = num_images - processed
                images = images[:batch_size]
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # Plot each image with prediction only
            for i in range(batch_size):
                img = images[i].cpu()
                # Unnormalize for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                img = img.clamp(0,1)
                img_np = img.permute(1,2,0).numpy()
                pred_label = predicted[i].item()
                pred_name = class_names[pred_label] if class_names else str(pred_label)
                plt.figure()
                plt.imshow(img_np)
                plt.title(f'Predicted: {pred_name}')
                plt.axis('off')
                # Save plot to disk
                os.makedirs('test_predictions', exist_ok=True)
                plot_path = f'test_predictions/img_{processed}_pred_{pred_name}.png'
                plt.savefig(plot_path)
                plt.close()
                processed += 1
                if num_images is not None and processed >= num_images:
                    break
            if num_images is not None and processed >= num_images:
                break
    if processed == 0:
        print('No test images processed.')
        return
    print(f'Inference complete. Saved {processed} prediction plots.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ResNet152 Training/Evaluation')
    parser.add_argument('--mode', choices=['finetune', 'train', 'test'], default='test', help='Mode to run: finetune, train, or test')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--num_images', type=int, default=10, help='Number of test images to evaluate')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()

    class_names = val_dataset.classes

    if args.mode == 'finetune':
        print('Fine-tuning last layer...')
        fine_tune_last_layer(resnet, train_loader, val_loader, criterion, num_epochs=args.epochs, patience=args.patience, save_path=args.save_path)
    elif args.mode == 'train':
        print('Training entire model...')
        train_entire_model(resnet, train_loader, val_loader, criterion, num_epochs=args.epochs, patience=args.patience, save_path=args.save_path)
    elif args.mode == 'test':
        print('Running inference on test images...')
        # Load saved model weights before inference
        if os.path.exists(args.save_path):
            resnet.load_state_dict(torch.load(args.save_path, map_location=device))
            print(f'Loaded model weights from {args.save_path}')
        else:
            print(f'Warning: Model weights file {args.save_path} not found. Using current model weights.')
        evaluate_on_test(resnet, test_loader, num_images=args.num_images, class_names=class_names)

if __name__ == '__main__':
    main()
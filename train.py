import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from model import ViT
from dataset import get_dataloaders
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torchvision.transforms as transforms
import matplotlib as mpl

def plot_sample_predictions(model, test_loader, device, save_dir, num_samples=8):
    """Plot sample image classification results"""
    model.eval()
    
    # Get a batch of data
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.sigmoid(outputs)
    
    # Create denormalization transform
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Set up image grid
    fig = plt.figure(figsize=(16, 8))
    for idx in range(num_samples):
        ax = plt.subplot(2, 4, idx + 1)
        
        # Denormalize image
        img = inv_normalize(images[idx])
        img = torch.clamp(img, 0, 1)
        
        # Display image
        plt.imshow(img.permute(1, 2, 0).cpu())
        
        # Get true label and prediction
        true_label = "Malignant" if labels[idx].item() == 1 else "Benign"
        pred_prob = probabilities[idx].item()
        pred_label = "Malignant" if pred_prob > 0.5 else "Benign"
        
        # Set title color (green for correct, red for incorrect)
        color = 'green' if (pred_prob > 0.5) == labels[idx].item() else 'red'
        
        # Add title
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {pred_prob:.2f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, save_dir):
    """Plot training and testing curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Set axis labels
    plt.gca().set_xticklabels(['Benign', 'Malignant'])
    plt.gca().set_yticklabels(['Benign', 'Malignant'])
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, save_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_with_predictions(model, test_loader, criterion, device):
    """Evaluate model and return predictions"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            
            scores = torch.sigmoid(outputs)
            predicted = (scores > 0.5).float()
            
            total_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    return (total_loss/len(test_loader), correct/total, 
            np.array(all_targets), np.array(all_predictions), np.array(all_scores))

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float().unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs.data) > 0.5).float()
        total += targets.size(0)
        correct += predicted.eq(targets.data.view_as(predicted)).sum().item()
        
        progress_bar.set_postfix({
            'Loss': total_loss/(batch_idx+1),
            'Acc': 100.*correct/total
        })
    
    return total_loss/len(train_loader), correct/total

def main():
    parser = argparse.ArgumentParser(description='Train ViT for Skin Cancer Classification')
    parser.add_argument('--data_dir', type=str, default='kaggle/input/segmented-images-of-the-skin-cancer-dataset',
                        help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training')
    args = parser.parse_args()
    
    # Create directory to save results
    results_dir = 'results1'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs('checkpoints1', exist_ok=True)
    
    # Create model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ).to(args.device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    
    # For recording training process
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        
        # Evaluate
        test_loss, test_acc, y_true, y_pred, y_scores = evaluate_with_predictions(
            model, test_loader, criterion, args.device
        )
        
        # Record results
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            print('Saving best model...')
            torch.save(model.state_dict(), 'checkpoints1/best_model.pth')
            best_acc = test_acc
            
            # Plot confusion matrix and ROC curve for best model
            plot_confusion_matrix(y_true, y_pred, results_dir)
            plot_roc_curve(y_true, y_scores, results_dir)
            
            # Plot sample predictions
            plot_sample_predictions(model, test_loader, args.device, results_dir)
    
    # Plot training process curves
    plot_training_curves(train_losses, test_losses, train_accs, test_accs, results_dir)

if __name__ == '__main__':
    main() 
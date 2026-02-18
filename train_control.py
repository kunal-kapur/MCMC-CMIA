"""
Control model training script for membership inference attacks.
Trains a ResNet model on a specified percentage of the training dataset
and saves which exact data points were used.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import ResNet18
import numpy as np
from datetime import datetime


def setup_device():
    """Setup GPU/CPU device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    return device


def load_datasets(data_percentage=100, seed=42):
    """
    Load CIFAR-10 dataset and sample a percentage of training data.
    
    Args:
        data_percentage: Percentage of training data to use (1-100)
        seed: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        train_indices: Indices of training samples used
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, 
                               transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, 
                              transform=transform_test)
    
    # Sample training data
    total_train_size = len(trainset)
    num_samples = max(1, int(total_train_size * data_percentage / 100))
    
    all_indices = np.arange(total_train_size)
    train_indices = np.random.choice(all_indices, size=num_samples, replace=False)
    train_indices = sorted(train_indices.tolist())
    
    # Create subset with sampled indices
    train_subset = Subset(trainset, train_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, 
                             num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, 
                            num_workers=2)
    
    return train_loader, test_loader, train_indices


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}]: Loss={train_loss/(batch_idx+1):.3f}, '
                  f'Acc={100.*correct/total:.2f}%')
    
    return train_loss / (batch_idx + 1), 100. * correct / total


def test(model, test_loader, criterion, device):
    """Test the model."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def save_training_metadata(save_dir, data_percentage, train_indices, 
                          test_accuracy, test_loss, num_epochs, seed):
    """
    Save metadata about the training run including which data was used.
    
    Args:
        save_dir: Directory to save metadata
        data_percentage: Percentage of data used
        train_indices: List of training sample indices used
        test_accuracy: Final test accuracy
        test_loss: Final test loss
        num_epochs: Number of epochs trained
        seed: Random seed used
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_percentage": data_percentage,
        "num_samples_used": len(train_indices),
        "total_cifar10_train_size": 50000,
        "train_indices": train_indices,  # EXACT data points used
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "num_epochs": num_epochs,
        "seed": seed,
        "model": "ResNet18"
    }
    
    # Save as JSON
    metadata_path = os.path.join(save_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining metadata saved to: {metadata_path}")
    print(f"\nSummary:")
    print(f"  - Used {len(train_indices)} samples ({data_percentage}% of training set)")
    print(f"  - Final test accuracy: {test_accuracy:.2f}%")
    print(f"  - Final test loss: {test_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train a control ResNet model')
    parser.add_argument('--data-percentage', type=float, default=100,
                       help='Percentage of training data to use (1-100)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save model and metadata')
    parser.add_argument('--model-name', type=str, default='control_model',
                       help='Name of the model checkpoint')
    
    args = parser.parse_args()
    
    if not (1 <= args.data_percentage <= 100):
        raise ValueError("data_percentage must be between 1 and 100")
    
    # Setup
    device = setup_device()
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\nLoading datasets (using {args.data_percentage}% of training data)...")
    train_loader, test_loader, train_indices = load_datasets(args.data_percentage, 
                                                              args.seed)
    print(f"Training on {len(train_indices)} samples")
    
    print("\nCreating ResNet18 model...")
    model = ResNet18()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                         weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nTraining for {args.epochs} epochs...\n")
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            model_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (accuracy: {test_acc:.2f}%)")
    
    final_model_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*60}")
    save_training_metadata(args.save_dir, args.data_percentage, train_indices,
                          best_test_acc, test_loss, args.epochs, args.seed)
    print(f"{'='*60}")
    
    print(f"\nModel checkpoints saved to: {args.save_dir}")
    print(f"  - Best model: {args.model_name}_best.pth")
    print(f"  - Final model: {args.model_name}_final.pth")


if __name__ == "__main__":
    main()

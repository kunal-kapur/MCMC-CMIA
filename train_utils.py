"""
Shared training utilities for ResNet models.
Provides unified training functions and configuration management to ensure
shadow models use the same training configuration as the target model.
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from global_variables import DATA_DIR, TRANSFORM_TRAIN, TRANSFORM_TEST


class TrainingConfig:
    """Container for training configuration parameters."""
    
    def __init__(self, 
                 epochs=200,
                 optimizer_type='adam',
                 lr=0.001,
                 momentum=0.9,  # Only used if optimizer_type is 'sgd'
                 weight_decay=5e-4,
                 scheduler_type='cosine',
                 batch_size=128,
                 num_workers=2):
        self.epochs = epochs
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def to_dict(self):
        """Convert config to dictionary for JSON serialization."""
        return {
            'epochs': self.epochs,
            'optimizer_type': self.optimizer_type,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'scheduler_type': self.scheduler_type,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_metadata(cls, metadata_path):
        """Load training config from metadata JSON file."""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'training_config' not in metadata:
            raise ValueError(
                f"No 'training_config' found in metadata file: {metadata_path}\n"
                "This may be an old metadata file. Please retrain the target model."
            )
        
        return cls.from_dict(metadata['training_config'])


def create_optimizer(model, config):
    """Create optimizer based on config."""
    if config.optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    if config.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs
        )
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=0.1
        )
    elif config.scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")


def create_dataloader(indices, batch_size, num_workers, train=True, shuffle=True):
    """Create a DataLoader for specified indices."""
    transform = TRANSFORM_TRAIN if train else TRANSFORM_TEST
    dataset = datasets.CIFAR10(
        root=DATA_DIR, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    if indices is not None:
        dataset = Subset(dataset, indices)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )


def train_one_epoch(model, train_loader, criterion, optimizer, device, verbose=False):
    """Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        verbose: Whether to print batch progress
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
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
        
        if verbose and (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}]: Loss={train_loss/(batch_idx+1):.3f}, '
                  f'Acc={100.*correct/total:.2f}%')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test/validation data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: torch device
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
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
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_indices, config, device, verbose=True, 
                test_loader=None, save_best_callback=None):
    """
    Complete training loop for a model.
    
    Args:
        model: Neural network model (already on device)
        train_indices: List of training sample indices
        config: TrainingConfig object
        device: torch device
        verbose: Whether to print epoch progress
        test_loader: Optional test DataLoader for validation
        save_best_callback: Optional function(epoch, test_acc) called when best model is achieved
    
    Returns:
        Tuple of (final_train_loss, final_train_acc, final_test_loss, final_test_acc)
        If test_loader is None, test metrics will be None.
    """
    # Create data loader
    train_loader = create_dataloader(
        train_indices, 
        config.batch_size, 
        config.num_workers,
        train=True,
        shuffle=True
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Training loop
    best_test_acc = 0
    final_test_loss = None
    final_test_acc = None
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            verbose=(verbose and epoch == 0)  # Only show batch details for first epoch
        )
        
        # Evaluate on test set if provided
        if test_loader is not None:
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            final_test_loss = test_loss
            final_test_acc = test_acc
            
            if verbose:
                print(f"Epoch [{epoch+1}/{config.epochs}] - "
                      f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            # Track best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if save_best_callback is not None:
                    save_best_callback(epoch + 1, test_acc)
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.epochs}] - Train Acc: {train_acc:.2f}%")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
    
    return train_loss, train_acc, final_test_loss, final_test_acc


def train_shadow_model(model, train_indices, config, device):
    """
    Train a shadow model (no test evaluation, no checkpointing).
    
    Args:
        model: Neural network model (already on device)
        train_indices: List of training sample indices
        config: TrainingConfig object
        device: torch device
    
    Returns:
        Trained model in eval mode
    """
    train_model(
        model, 
        train_indices, 
        config, 
        device, 
        verbose=False,
        test_loader=None,
        save_best_callback=None
    )
    
    model.eval()
    return model

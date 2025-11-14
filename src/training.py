import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from model import AttentionUNet, UNet


class ModelManager:
    """
    Utility class for model management, saving, and loading
    """
    def __init__(self, save_dir="models", run_name=None):
        self.save_dir = save_dir
        self.run_name = run_name or "default"
        
        # Create main directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create run-specific subdirectories
        self.run_dir = os.path.join(save_dir, self.run_name)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.summaries_dir = os.path.join(self.run_dir, "summaries")
        self.exports_dir = os.path.join(self.run_dir, "exports")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        
        print(f"ModelManager initialized for run: {self.run_name}")
        print(f"Saving to: {self.run_dir}")
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_model_summary(self, model, input_size=(1, 1, 512, 512), save_summary=True):
        """
        Generate and optionally save model summary
        """
        summary_info = {
            "model_name": model.__class__.__name__,
            "run_name": self.run_name,
            "total_parameters": self.count_parameters(model),
            "input_size": input_size,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Model: {summary_info['model_name']}")
        print(f"Run: {summary_info['run_name']}")
        print(f"Total trainable parameters: {summary_info['total_parameters']:,}")
        
        # Test with dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            summary_info["output_shape"] = list(output.shape)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        
        # Save summary if requested
        if save_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(
                self.summaries_dir, 
                f"{summary_info['model_name']}_{self.run_name}_summary_{timestamp}.json"
            )
            
            with open(summary_file, 'w') as f:
                json.dump(summary_info, f, indent=2)
            
            print(f"Model summary saved: {summary_file}")
        
        return summary_info
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, 
                       is_best=False, metadata=None):
        """
        Save model checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_class': model.__class__.__name__,
            'run_name': self.run_name,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"checkpoint_{self.run_name}_epoch_{epoch}_{timestamp}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, f"best_model_{self.run_name}.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def export_model(self, model, model_name=None, metadata=None):
        """
        Export model for inference (pickle format for easy loading)
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare export data
        export_data = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'run_name': self.run_name,
            'model_config': {
                'n_channels': getattr(model, 'n_channels', 1),
                'n_classes': getattr(model, 'n_classes', 1),
                'bilinear': getattr(model, 'bilinear', False)
            },
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Save as pickle for easy loading
        pickle_path = os.path.join(
            self.exports_dir, 
            f"{model_name}_{self.run_name}_export_{timestamp}.pkl"
        )
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        # Also save model state dict as .pth
        pth_path = os.path.join(
            self.exports_dir, 
            f"{model_name}_{self.run_name}_state_dict_{timestamp}.pth"
        )
        torch.save(export_data['model_state_dict'], pth_path)
        
        print(f"Model exported:")
        print(f"  Pickle: {pickle_path}")
        print(f"  State Dict: {pth_path}")
        
        return pickle_path, pth_path
    
    def load_model_from_pickle(self, pickle_path):
        """
        Load model from pickle file
        """
        with open(pickle_path, 'rb') as f:
            export_data = pickle.load(f)
        
        model = export_data['model']
        print(f"Model loaded from: {pickle_path}")
        return model, export_data['metadata']
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """
        Load model from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('train_loss', 0)
        val_loss = checkpoint.get('val_loss', 0)
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return epoch, train_loss, val_loss


class MicroscopyDataset(Dataset):
    """
    Custom dataset for loading microscopy images and binary masks
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Verify that images and masks match
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        
        # Normalize mask to 0-1 range
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)  # Ensure binary
        
        # Apply transforms if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension if not present (for grayscale)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

class DataManager:
    """
    Manages data loading, splitting, and preparation for training
    """
    def __init__(self, cropped_dir, binary_dir, image_size=(512, 512), use_augmented=False):
        self.cropped_dir = cropped_dir
        self.binary_dir = binary_dir
        self.image_size = image_size
        self.use_augmented = use_augmented
        
        # Define transforms
        self.train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])
    
    def get_image_mask_pairs(self):
        """
        Find all matching image-mask pairs
        """
        # Find all cropped images
        cropped_files = glob.glob(os.path.join(self.cropped_dir, "*.cropped.tif"))
        
        image_paths = []
        mask_paths = []
        
        for cropped_file in cropped_files:
            # Skip augmented files if not using augmented data
            base_name = os.path.basename(cropped_file)
            if not self.use_augmented and '_aug_' in base_name:
                continue
            
            # Get base name and construct mask path
            base_name = base_name.replace('.cropped.tif', '')
            mask_file = os.path.join(self.binary_dir, f"{base_name}.binary.tif")
            
            # Check if mask exists
            if os.path.exists(mask_file):
                image_paths.append(cropped_file)
                mask_paths.append(mask_file)
            else:
                print(f"Warning: No mask found for {cropped_file}")
        
        print(f"Found {len(image_paths)} image-mask pairs")
        if not self.use_augmented:
            print("Using only original (non-augmented) data")
        else:
            print("Using all data including augmented versions")
        
        return image_paths, mask_paths
    
    def create_train_val_split(self, test_size=0.2, random_state=42):
        """
        Split data ensuring no leakage between original and augmented versions
        """
        image_paths, mask_paths = self.get_image_mask_pairs()
        
        if self.use_augmented:
            # Group by original image (remove augmentation suffixes)
            original_groups = {}
            for img_path, mask_path in zip(image_paths, mask_paths):
                base_name = os.path.basename(img_path)
                # Remove augmentation suffix (_aug_0, _aug_1, etc.)
                original_name = base_name.split('_aug_')[0]
                
                if original_name not in original_groups:
                    original_groups[original_name] = {'images': [], 'masks': []}
                
                original_groups[original_name]['images'].append(img_path)
                original_groups[original_name]['masks'].append(mask_path)
            
            # Split by original images
            original_names = list(original_groups.keys())
            train_originals, val_originals = train_test_split(
                original_names, test_size=test_size, random_state=random_state
            )
            
            print(f"Training on {len(train_originals)} original images")
            print(f"Validating on {len(val_originals)} original images")
            
            # Collect all images/masks for each split
            train_images, train_masks = [], []
            val_images, val_masks = [], []
            
            for orig in train_originals:
                train_images.extend(original_groups[orig]['images'])
                train_masks.extend(original_groups[orig]['masks'])
            
            for orig in val_originals:
                val_images.extend(original_groups[orig]['images'])
                val_masks.extend(original_groups[orig]['masks'])
        
        else:
            # Simple split for original data only
            train_images, val_images, train_masks, val_masks = train_test_split(
                image_paths, mask_paths, test_size=test_size, random_state=random_state
            )
            
            print(f"Training on {len(train_images)} original images")
            print(f"Validating on {len(val_images)} original images")
        
        print(f"Total training samples: {len(train_images)}")
        print(f"Total validation samples: {len(val_images)}")
        
        return train_images, val_images, train_masks, val_masks
    
    def create_dataloaders(self, batch_size=8, num_workers=4, test_size=0.2):
        """
        Create PyTorch DataLoaders for training and validation
        """
        train_images, val_images, train_masks, val_masks = self.create_train_val_split(test_size)
        
        # Create datasets
        train_dataset = MicroscopyDataset(
            train_images, train_masks, 
            transform=self.train_transform
        )
        
        val_dataset = MicroscopyDataset(
            val_images, val_masks, 
            transform=self.val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def visualize_batch(self, dataloader, num_samples=4):
        """
        Visualize a batch of images and masks
        """
        # Get a batch
        images, masks = next(iter(dataloader))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i in range(min(num_samples, len(images))):
            # Convert tensor to numpy for visualization
            img = images[i].squeeze().numpy()
            mask = masks[i].squeeze().numpy()
            
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Mask {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

class LossFunction:
    """
    Custom loss functions for segmentation
    """
    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        """
        Dice loss for segmentation
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    @staticmethod
    def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
        """
        Combination of Binary Cross Entropy and Dice Loss
        """
        bce = nn.BCEWithLogitsLoss()(pred, target)
        dice = LossFunction.dice_loss(pred, target)
        
        return bce_weight * bce + dice_weight * dice

class Trainer:
    """
    Training manager for UNet model
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', save_dir="models", run_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_name = run_name or "default"
        
        # Initialize model manager with run name
        self.model_manager = ModelManager(save_dir, run_name=self.run_name)
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function
        self.criterion = LossFunction.combined_loss
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Generate and save model summary
        print("="*60)
        print("MODEL SUMMARY")
        print("="*60)
        self.model_summary = self.model_manager.get_model_summary(
            self.model, 
            input_size=(1, 1, 512, 512), 
            save_summary=True
        )
    
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for images, masks in pbar:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs=100):
        """
        Full training loop with enhanced saving
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Run name: {self.run_name}")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        training_metadata = {
            'run_name': self.run_name,
            'num_epochs': num_epochs,
            'batch_size': self.train_loader.batch_size,
            'device': str(self.device),
            'optimizer': self.optimizer.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
            'loss_function': 'Combined BCE + Dice Loss'
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint every 10 epochs or if best
            if (epoch + 1) % 10 == 0 or is_best:
                self.model_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    is_best=is_best,
                    metadata=training_metadata
                )
        
        print("Training completed!")
        
        # Export final model
        final_metadata = {
            **training_metadata,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'best_val_loss': self.best_val_loss,
            'total_epochs_trained': num_epochs
        }
        
        pickle_path, pth_path = self.model_manager.export_model(
            self.model, 
            metadata=final_metadata
        )
        
        # Plot and save losses
        self.plot_losses(save=True)
        
        return pickle_path, pth_path
    
    def plot_losses(self, save=False):
        """
        Plot training and validation losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses - {self.run_name}')
        plt.legend()
        plt.grid(True)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(
                self.model_manager.summaries_dir, 
                f"training_losses_{self.run_name}_{timestamp}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved: {plot_path}")
        
        plt.show()

def main_original():
    """
    Training function for original data only
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    cropped_dir = "/home/bhunn1/vision_analysis/src/data/cropped"
    binary_dir = "/home/bhunn1/vision_analysis/src/data/binary"
    
    # Create data manager - ONLY ORIGINAL DATA
    data_manager = DataManager(
        cropped_dir, 
        binary_dir, 
        image_size=(512, 512),
        use_augmented=False
    )
    
    # Create dataloaders
    train_loader, val_loader = data_manager.create_dataloaders(
        batch_size=4,
        num_workers=4,
        test_size=0.2
    )
    
    # Visualize some data
    print("Visualizing training data...")
    data_manager.visualize_batch(train_loader)
    
    # Initialize AttentionUNet model
    print("\n" + "="*60)
    print("INITIALIZING ATTENTION UNET MODEL - ORIGINAL DATA")
    print("="*60)
    
    model = AttentionUNet(n_channels=1, n_classes=1)
    
    # Initialize trainer with specific run name
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        device, 
        save_dir="/home/bhunn1/vision_analysis/models",
        run_name="original_data"
    )
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING - ORIGINAL DATA ONLY")
    print("="*60)
    
    pickle_path, pth_path = trainer.train(num_epochs=50)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final model saved as:")
    print(f"  Pickle: {pickle_path}")
    print(f"  State Dict: {pth_path}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

def main_augmented():
    """
    Training function for augmented data
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    cropped_dir = "/home/bhunn1/vision_analysis/src/data/cropped"
    binary_dir = "/home/bhunn1/vision_analysis/src/data/binary"
    
    # Create data manager - WITH AUGMENTED DATA
    data_manager = DataManager(
        cropped_dir, 
        binary_dir, 
        image_size=(512, 512),
        use_augmented=True  # Set to True for augmented data
    )
    
    # Create dataloaders
    train_loader, val_loader = data_manager.create_dataloaders(
        batch_size=4,
        num_workers=4,
        test_size=0.2
    )
    
    # Visualize some data
    print("Visualizing training data...")
    data_manager.visualize_batch(train_loader)
    
    # Initialize AttentionUNet model
    print("\n" + "="*60)
    print("INITIALIZING ATTENTION UNET MODEL - AUGMENTED DATA")
    print("="*60)
    
    model = AttentionUNet(n_channels=1, n_classes=1)
    
    # Initialize trainer with specific run name
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        device, 
        save_dir="/home/bhunn1/vision_analysis/models",
        run_name="augmented_data"
    )
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING - WITH AUGMENTED DATA")
    print("="*60)
    
    pickle_path, pth_path = trainer.train(num_epochs=50)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final model saved as:")
    print(f"  Pickle: {pickle_path}")
    print(f"  State Dict: {pth_path}")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

def main():
    """
    Main function - choose which training to run
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "original":
            main_original()
        elif sys.argv[1] == "augmented":
            main_augmented()
        else:
            print("Usage: python training.py [original|augmented]")
            print("Running default (original) training...")
            main_original()
    else:
        # Default to augmented training since you want to train with augmented data now
        main_augmented()

if __name__ == "__main__":
    main()
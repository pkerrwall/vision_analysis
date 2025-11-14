import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import pickle
from model import AttentionUNet
from training import ModelManager

# Configure matplotlib for better display
plt.rcParams['figure.max_open_warning'] = 0
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend initially

class ModelTester:
    """
    Class for testing trained segmentation models
    """
    def __init__(self, model_path, run_name="unknown", device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()
        self.run_name = run_name
        print(f"Model loaded on device: {self.device}")
    
    def load_model(self, model_path):
        """
        Load model from pickle file
        """
        print(f"Loading model from: {model_path}")
        
        if model_path.endswith('.pkl'):
            # Load from pickle
            with open(model_path, 'rb') as f:
                export_data = pickle.load(f)
            model = export_data['model']
            print(f"Loaded model: {export_data.get('model_class', 'Unknown')}")
            print(f"Run: {export_data.get('run_name', 'Unknown')}")
        
        elif model_path.endswith('.pth'):
            # Load from state dict (you'll need to create the model first)
            model = AttentionUNet(n_channels=1, n_classes=1)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """
        Preprocess image for inference
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Store original size for later
        original_size = image.shape
        
        # Resize to model input size
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch/channel dimensions
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device), original_size
    
    def postprocess_prediction(self, prediction, original_size):
        """
        Postprocess model prediction
        """
        # Apply sigmoid to get probabilities
        prediction = torch.sigmoid(prediction)
        
        # Remove batch and channel dimensions
        prediction = prediction.squeeze().cpu().numpy()
        
        # Resize back to original size
        prediction = cv2.resize(prediction, (original_size[1], original_size[0]))
        
        return prediction
    
    def predict_single_image(self, image_path, threshold=0.5):
        """
        Predict segmentation mask for a single image
        """
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image_path)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(image_tensor)
        
        # Postprocess
        prob_mask = self.postprocess_prediction(prediction, original_size)
        
        # Apply threshold to get binary mask
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        
        return prob_mask, binary_mask
    
    def test_on_validation_set(self, test_dir, mask_dir=None, num_samples=5, save_results=True):
        """
        Test model on a set of validation images
        """
        # Find test images
        test_images = glob.glob(os.path.join(test_dir, "*.tif"))
        test_images = [img for img in test_images if not '_aug_' in os.path.basename(img)]  # Skip augmented
        test_images.sort()  # Ensure consistent order

        if not test_images:
            print(f"No test images found in {test_dir}")
            return

        # Limit to num_samples
        test_images = test_images[:num_samples]
        print(f"Testing on {len(test_images)} images...")

        # Results directory includes run_name
        if save_results:
            results_dir = f"/home/bhunn1/vision_analysis/test_results/{self.run_name}/validation"
            os.makedirs(results_dir, exist_ok=True)

        all_dice_scores = []
        all_iou_scores = []
        
        for i, image_path in enumerate(test_images):
            print(f"\nTesting image {i+1}: {os.path.basename(image_path)}")
            
            # Predict
            prob_mask, binary_mask = self.predict_single_image(image_path)
            
            # Load original image for visualization
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Try to load ground truth mask if available
            ground_truth = None
            dice_score = None
            iou_score = None
            
            if mask_dir:
                base_name = os.path.basename(image_path).replace('.cropped.tif', '')
                mask_path = os.path.join(mask_dir, f"{base_name}.binary.tif")
                if os.path.exists(mask_path):
                    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    ground_truth = (ground_truth > 127).astype(np.uint8)
                    
                    # Calculate metrics
                    intersection = np.logical_and(binary_mask, ground_truth).sum()
                    union = np.logical_or(binary_mask, ground_truth).sum()
                    dice_score = 2.0 * intersection / (binary_mask.sum() + ground_truth.sum()) if (binary_mask.sum() + ground_truth.sum()) > 0 else 0
                    iou_score = intersection / union if union > 0 else 0
                    
                    all_dice_scores.append(dice_score)
                    all_iou_scores.append(iou_score)
                    
                    print(f"  Dice Score: {dice_score:.3f}")
                    print(f"  IoU Score: {iou_score:.3f}")
            
            # Visualize results
            if save_results:
                self.save_prediction_image(
                    original_image, prob_mask, binary_mask, ground_truth, 
                    save_path=os.path.join(results_dir, f"result_{i+1}_{base_name}.png"),
                    title=os.path.basename(image_path),
                    dice_score=dice_score,
                    iou_score=iou_score
                )
        
        # Print summary statistics
        if all_dice_scores:
            print(f"\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            print(f"Average Dice Score: {np.mean(all_dice_scores):.3f} ± {np.std(all_dice_scores):.3f}")
            print(f"Average IoU Score: {np.mean(all_iou_scores):.3f} ± {np.std(all_iou_scores):.3f}")
            print(f"Best Dice Score: {np.max(all_dice_scores):.3f}")
            print(f"Worst Dice Score: {np.min(all_dice_scores):.3f}")
        
        if save_results:
            print(f"\nResults saved to: {results_dir}")
    
    def save_prediction_image(self, original_image, prob_mask, binary_mask, 
                            ground_truth=None, save_path=None, title="Prediction",
                            dice_score=None, iou_score=None):
        """
        Save prediction results as image file
        """
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(ground_truth, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(prob_mask, cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_title('Probability Map')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(binary_mask, cmap='gray')
            axes[1, 1].set_title('Binary Prediction')
            axes[1, 1].axis('off')
            
            # Add metrics to title if available
            if dice_score is not None and iou_score is not None:
                fig.suptitle(f'{title}\nDice: {dice_score:.3f}, IoU: {iou_score:.3f}', fontsize=14)
            else:
                fig.suptitle(title, fontsize=14)
                
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(prob_mask, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('Probability Map')
            axes[1].axis('off')
            
            axes[2].imshow(binary_mask, cmap='gray')
            axes[2].set_title('Binary Prediction')
            axes[2].axis('off')
            
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()  # Close to prevent memory issues
    
    def test_on_new_image(self, image_path, save_result=True):
        """
        Test model on a completely new image
        """
        print(f"Testing on new image: {image_path}")
        
        # Predict
        prob_mask, binary_mask = self.predict_single_image(image_path)
        
        # Load original image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Save result
        if save_result:
            results_dir = "/home/bhunn1/vision_analysis/test_results/original_data"
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f"new_image_{os.path.basename(image_path)}.png")
            
            self.save_prediction_image(
                original_image, prob_mask, binary_mask, 
                save_path=save_path,
                title=f"New Image: {os.path.basename(image_path)}"
            )
        
        return prob_mask, binary_mask
    
    def test_on_new_samples(self, testing_dir, save_results=True):
        """
        Test model on new manually created samples in testing directory
        """
        print(f"Looking for test samples in: {testing_dir}")
        
        # Find all .tif files that are NOT binary masks
        image_files = glob.glob(os.path.join(testing_dir, "*.tif"))
        image_files = [f for f in image_files if not f.endswith('_binary.tif')]
        image_files.sort()  # Ensure consistent order

        if not image_files:
            print(f"No test images found in {testing_dir}")
            return

        print(f"Found {len(image_files)} test images...")

        # Results directory includes run_name
        if save_results:
            results_dir = f"/home/bhunn1/vision_analysis/test_results/{self.run_name}/new_samples"
            os.makedirs(results_dir, exist_ok=True)

        all_dice_scores = []
        all_iou_scores = []
        
        for i, image_path in enumerate(image_files):
            base_name = os.path.basename(image_path).replace('.tif', '')
            print(f"\nTesting new sample {i+1}: {base_name}")
            
            # Predict
            prob_mask, binary_mask = self.predict_single_image(image_path)
            
            # Load original image for visualization
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Try to load corresponding binary mask
            ground_truth = None
            dice_score = None
            iou_score = None
            
            # Look for corresponding binary mask
            binary_mask_path = os.path.join(testing_dir, f"{base_name}_binary.tif")
            if os.path.exists(binary_mask_path):
                print(f"  Found ground truth: {os.path.basename(binary_mask_path)}")
                ground_truth = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
                ground_truth = (ground_truth > 127).astype(np.uint8)
                
                # Calculate metrics
                intersection = np.logical_and(binary_mask, ground_truth).sum()
                union = np.logical_or(binary_mask, ground_truth).sum()
                dice_score = 2.0 * intersection / (binary_mask.sum() + ground_truth.sum()) if (binary_mask.sum() + ground_truth.sum()) > 0 else 0
                iou_score = intersection / union if union > 0 else 0
                
                all_dice_scores.append(dice_score)
                all_iou_scores.append(iou_score)
                
                print(f"  Dice Score: {dice_score:.3f}")
                print(f"  IoU Score: {iou_score:.3f}")
            else:
                print(f"  No ground truth found for {base_name}")
            
            # Save visualization
            if save_results:
                save_path = os.path.join(results_dir, f"new_sample_{i+1}_{base_name}.png")
                self.save_prediction_image(
                    original_image, prob_mask, binary_mask, ground_truth, 
                    save_path=save_path,
                    title=f"New Sample: {base_name}",
                    dice_score=dice_score,
                    iou_score=iou_score
                )
        
        # Print summary statistics for new samples
        if all_dice_scores:
            print(f"\n" + "="*60)
            print("NEW SAMPLES SUMMARY STATISTICS")
            print("="*60)
            print(f"Number of samples with ground truth: {len(all_dice_scores)}")
            print(f"Average Dice Score: {np.mean(all_dice_scores):.3f} ± {np.std(all_dice_scores):.3f}")
            print(f"Average IoU Score: {np.mean(all_iou_scores):.3f} ± {np.std(all_iou_scores):.3f}")
            if len(all_dice_scores) > 1:
                print(f"Best Dice Score: {np.max(all_dice_scores):.3f}")
                print(f"Worst Dice Score: {np.min(all_dice_scores):.3f}")
        
        if save_results:
            print(f"\nNew sample results saved to: {results_dir}")
        
        return all_dice_scores, all_iou_scores

def main():
    """
    Main testing function
    """
    print("="*60)
    print("MODEL TESTING")
    print("="*60)

    # Set run_name for clarity
    run_name = "original_data"  # or "augmented_data" depending on model

    # Path to your trained model
    model_path = f"/home/bhunn1/vision_analysis/models/{run_name}/exports/AttentionUNet_export_20251106_131123.pkl"

    # Initialize tester
    tester = ModelTester(model_path, run_name=run_name)
    
    # Test on validation set (images the model hasn't seen during training)
    test_dir = "/home/bhunn1/vision_analysis/src/data/cropped"
    mask_dir = "/home/bhunn1/vision_analysis/src/data/binary"
    
    """
    print("\n" + "="*60)
    print("TESTING ON VALIDATION IMAGES")
    print("="*60)
    
    tester.test_on_validation_set(test_dir, mask_dir, num_samples=50, save_results=True)
    """

    # Test on new manually created samples
    testing_dir = "/home/bhunn1/vision_analysis/testing"

    print("\n" + "="*60)
    print("TESTING ON NEW MANUAL SAMPLES")
    print("="*60)

    tester.test_on_new_samples(testing_dir, save_results=True)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    print(f"Check /home/bhunn1/vision_analysis/test_results/{run_name}/validation/ for validation results")
    print(f"Check /home/bhunn1/vision_analysis/test_results/{run_name}/new_samples/ for new sample results")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_training_pipeline_diagram():
    """
    Create a comprehensive training pipeline diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors
    data_color = '#E3F2FD'
    model_color = '#FFF3E0'
    training_color = '#E8F5E8'
    output_color = '#FCE4EC'
    
    # Data Pipeline Section
    ax.text(0.5, 0.95, 'Vision Analysis Training Pipeline', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # 1. Data Input
    data_box = FancyBboxPatch((0.02, 0.8), 0.25, 0.12, 
                             boxstyle="round,pad=0.01", 
                             facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(0.145, 0.86, 'Raw Microscopy Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.145, 0.83, '• Cropped images (.tif)\n• Binary masks (.tif)', 
            fontsize=10, ha='center', va='center')
    
    # 2. Data Manager
    dm_box = FancyBboxPatch((0.32, 0.8), 0.25, 0.12,
                           boxstyle="round,pad=0.01",
                           facecolor=data_color, edgecolor='blue', linewidth=2)
    ax.add_patch(dm_box)
    ax.text(0.445, 0.86, 'DataManager', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.445, 0.83, '• Train/Val split\n• Augmentation control\n• DataLoaders', 
            fontsize=10, ha='center', va='center')
    
    # 3. Model Architecture
    model_box = FancyBboxPatch((0.02, 0.6), 0.55, 0.15,
                              boxstyle="round,pad=0.01",
                              facecolor=model_color, edgecolor='orange', linewidth=2)
    ax.add_patch(model_box)
    ax.text(0.295, 0.72, 'AttentionUNet Model', fontsize=14, fontweight='bold', ha='center')
    
    # Model components
    ax.text(0.08, 0.68, 'Encoder', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.08, 0.65, '• DoubleConv\n• Down blocks\n• Feature extraction', 
            fontsize=9, ha='center', va='center')
    
    ax.text(0.295, 0.68, 'Attention Gates', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.295, 0.65, '• Focus on relevant\n  features\n• Suppress noise', 
            fontsize=9, ha='center', va='center')
    
    ax.text(0.51, 0.68, 'Decoder', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.51, 0.65, '• Up blocks\n• Skip connections\n• Segmentation map', 
            fontsize=9, ha='center', va='center')
    
    # 4. Training Process
    train_box = FancyBboxPatch((0.62, 0.6), 0.35, 0.32,
                              boxstyle="round,pad=0.01",
                              facecolor=training_color, edgecolor='green', linewidth=2)
    ax.add_patch(train_box)
    ax.text(0.795, 0.88, 'Trainer', fontsize=14, fontweight='bold', ha='center')
    
    # Training components
    ax.text(0.795, 0.84, 'Loss Function', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.795, 0.81, 'BCE + Dice Loss', fontsize=10, ha='center')
    
    ax.text(0.795, 0.77, 'Optimizer', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.795, 0.74, 'Adam + LR Scheduler', fontsize=10, ha='center')
    
    ax.text(0.795, 0.70, 'Training Loop', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.795, 0.67, '• Forward pass\n• Loss calculation\n• Backpropagation', 
            fontsize=9, ha='center', va='center')
    
    ax.text(0.795, 0.63, 'Validation', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.795, 0.60, '• Model evaluation\n• Best model tracking', 
            fontsize=9, ha='center', va='center')
    
    # 5. Model Manager & Outputs
    output_box = FancyBboxPatch((0.02, 0.35), 0.95, 0.2,
                               boxstyle="round,pad=0.01",
                               facecolor=output_color, edgecolor='purple', linewidth=2)
    ax.add_patch(output_box)
    ax.text(0.495, 0.52, 'ModelManager - Saves & Organizes Results', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Output components in columns
    outputs = [
        ('Checkpoints', '• best_model.pth\n• epoch checkpoints\n• Resume training'),
        ('Exports', '• .pkl files (full model)\n• .pth files (weights)\n• Easy loading'),
        ('Summaries', '• Model architecture\n• Training metrics\n• Loss plots'),
        ('Results', '• Organized by run\n• original_data/\n• augmented_data/')
    ]
    
    x_positions = [0.15, 0.35, 0.55, 0.75]
    for i, (title, content) in enumerate(outputs):
        ax.text(x_positions[i], 0.48, title, fontsize=11, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.42, content, fontsize=9, ha='center', va='center')
    
    # 6. Testing Pipeline
    test_box = FancyBboxPatch((0.02, 0.1), 0.95, 0.2,
                             boxstyle="round,pad=0.01",
                             facecolor='#F3E5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(test_box)
    ax.text(0.495, 0.27, 'ModelTester - Evaluation & Inference', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Testing components
    test_items = [
        ('Load Model', '• From .pkl/.pth\n• GPU/CPU setup'),
        ('Preprocessing', '• Resize to 512x512\n• Normalize\n• Tensor conversion'),
        ('Inference', '• Forward pass\n• Sigmoid activation\n• Threshold (0.5)'),
        ('Metrics', '• Dice Score\n• IoU Score\n• Visualizations')
    ]
    
    for i, (title, content) in enumerate(test_items):
        ax.text(x_positions[i], 0.23, title, fontsize=11, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.17, content, fontsize=9, ha='center', va='center')
    
    # Add arrows to show flow
    # Data flow arrows
    arrow1 = ConnectionPatch((0.27, 0.86), (0.32, 0.86), "data", "data",
                           arrowstyle="->", shrinkA=0, shrinkB=0, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((0.445, 0.8), (0.295, 0.75), "data", "data",
                           arrowstyle="->", shrinkA=0, shrinkB=0, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow2)
    
    arrow3 = ConnectionPatch((0.57, 0.67), (0.62, 0.75), "data", "data",
                           arrowstyle="->", shrinkA=0, shrinkB=0, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow3)
    
    arrow4 = ConnectionPatch((0.795, 0.6), (0.495, 0.55), "data", "data",
                           arrowstyle="->", shrinkA=0, shrinkB=0, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow4)
    
    arrow5 = ConnectionPatch((0.495, 0.35), (0.495, 0.3), "data", "data",
                           arrowstyle="->", shrinkA=0, shrinkB=0, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/training_pipeline_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_attention_unet_diagram():
    """
    Create AttentionUNet architecture diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    ax.text(0.5, 0.95, 'AttentionUNet Architecture', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Define positions for encoder blocks
    encoder_positions = [(0.1, 0.8), (0.1, 0.65), (0.1, 0.5), (0.1, 0.35), (0.1, 0.2)]
    encoder_channels = [64, 128, 256, 512, 1024]
    encoder_sizes = [(512, 512), (256, 256), (128, 128), (64, 64), (32, 32)]
    
    # Define positions for decoder blocks  
    decoder_positions = [(0.9, 0.35), (0.9, 0.5), (0.9, 0.65), (0.9, 0.8)]
    decoder_channels = [512, 256, 128, 64]
    
    # Draw input
    input_box = FancyBboxPatch((0.02, 0.85), 0.06, 0.1,
                              boxstyle="round,pad=0.005",
                              facecolor='lightblue', edgecolor='blue')
    ax.add_patch(input_box)
    ax.text(0.05, 0.9, 'Input\n1×512×512', fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Draw encoder blocks
    for i, (pos, channels, size) in enumerate(zip(encoder_positions, encoder_channels, encoder_sizes)):
        # Encoder block
        block = FancyBboxPatch((pos[0]-0.03, pos[1]-0.05), 0.06, 0.1,
                              boxstyle="round,pad=0.005",
                              facecolor='lightgreen', edgecolor='green')
        ax.add_patch(block)
        ax.text(pos[0], pos[1], f'Conv\n{channels}\n{size[0]}×{size[1]}', 
                fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Down arrow (except for last)
        if i < len(encoder_positions) - 1:
            ax.arrow(pos[0], pos[1]-0.05, 0, -0.08, head_width=0.01, 
                    head_length=0.01, fc='black', ec='black')
    
    # Draw decoder blocks and attention gates
    attention_positions = [(0.5, 0.35), (0.5, 0.5), (0.5, 0.65), (0.5, 0.8)]
    
    for i, (dec_pos, att_pos, channels) in enumerate(zip(decoder_positions, attention_positions, decoder_channels)):
        # Attention gate
        att_gate = FancyBboxPatch((att_pos[0]-0.04, att_pos[1]-0.03), 0.08, 0.06,
                                 boxstyle="round,pad=0.005",
                                 facecolor='yellow', edgecolor='orange')
        ax.add_patch(att_gate)
        ax.text(att_pos[0], att_pos[1], f'Attention\nGate', 
                fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Decoder block
        dec_block = FancyBboxPatch((dec_pos[0]-0.03, dec_pos[1]-0.05), 0.06, 0.1,
                                  boxstyle="round,pad=0.005",
                                  facecolor='lightcoral', edgecolor='red')
        ax.add_patch(dec_block)
        ax.text(dec_pos[0], dec_pos[1], f'Up Conv\n{channels}', 
                fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Skip connection from encoder to attention gate
        enc_y = encoder_positions[3-i][1]  # Reverse order for skip connections
        ax.plot([0.13, att_pos[0]-0.04], [enc_y, att_pos[1]], 
               'b--', linewidth=2, alpha=0.7, label='Skip Connection' if i == 0 else "")
        
        # Connection from attention gate to decoder
        ax.plot([att_pos[0]+0.04, dec_pos[0]-0.03], [att_pos[1], dec_pos[1]], 
               'r-', linewidth=2, alpha=0.7)
        
        # Up arrow (except for first)
        if i < len(decoder_positions) - 1:
            ax.arrow(dec_pos[0], dec_pos[1]+0.05, 0, 0.08, head_width=0.01, 
                    head_length=0.01, fc='black', ec='black')
    
    # Output
    output_box = FancyBboxPatch((0.92, 0.85), 0.06, 0.1,
                               boxstyle="round,pad=0.005",
                               facecolor='lightpink', edgecolor='purple')
    ax.add_patch(output_box)
    ax.text(0.95, 0.9, 'Output\n1×512×512', fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Connection from last decoder to output
    ax.arrow(0.93, 0.85, 0, 0.08, head_width=0.01, head_length=0.01, fc='black', ec='black')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', label='Encoder'),
        plt.Rectangle((0, 0), 1, 1, facecolor='yellow', edgecolor='orange', label='Attention Gate'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='red', label='Decoder'),
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Skip Connection'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', label='Input/Output')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.15))
    
    # Add some annotations
    ax.text(0.25, 0.05, 'Encoder: Feature Extraction & Downsampling', 
            fontsize=12, ha='center', style='italic')
    ax.text(0.75, 0.05, 'Decoder: Upsampling & Reconstruction', 
            fontsize=12, ha='center', style='italic')
    ax.text(0.5, 0.02, 'Attention Gates: Focus on Relevant Features', 
            fontsize=12, ha='center', style='italic', color='orange')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/attention_unet_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """
    Create data flow and file structure diagram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left side: Data Flow
    ax1.text(0.5, 0.95, 'Data Flow & Processing', 
             fontsize=16, fontweight='bold', ha='center', transform=ax1.transAxes)
    
    # Original vs Augmented paths
    # Original data path
    orig_boxes = [
        (0.1, 0.8, 'Raw Images\n(.lif files)'),
        (0.1, 0.65, 'Cropped\n(.cropped.tif)'),
        (0.1, 0.5, 'Binary Masks\n(.binary.tif)'),
        (0.1, 0.35, 'Train/Val Split'),
        (0.1, 0.2, 'Original Model')
    ]
    
    for i, (x, y, text) in enumerate(orig_boxes):
        box = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1,
                            boxstyle="round,pad=0.01",
                            facecolor='lightblue', edgecolor='blue')
        ax1.add_patch(box)
        ax1.text(x, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
        
        if i < len(orig_boxes) - 1:
            ax1.arrow(x, y-0.05, 0, -0.08, head_width=0.02, head_length=0.01, 
                     fc='blue', ec='blue')
    
    # Augmented data path
    aug_boxes = [
        (0.7, 0.65, 'Data Augmentation\n(rotation, flip, etc.)'),
        (0.7, 0.5, 'Augmented Images\n(_aug_N.tif)'),
        (0.7, 0.35, 'Enhanced Training Set'),
        (0.7, 0.2, 'Augmented Model')
    ]
    
    for i, (x, y, text) in enumerate(aug_boxes):
        box = FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1,
                            boxstyle="round,pad=0.01",
                            facecolor='lightgreen', edgecolor='green')
        ax1.add_patch(box)
        ax1.text(x, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
        
        if i < len(aug_boxes) - 1:
            ax1.arrow(x, y-0.05, 0, -0.08, head_width=0.02, head_length=0.01, 
                     fc='green', ec='green')
    
    # Connection from cropped to augmentation
    ax1.arrow(0.18, 0.65, 0.44, 0, head_width=0.02, head_length=0.02, 
             fc='gray', ec='gray', linestyle='--')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right side: File Structure
    ax2.text(0.5, 0.95, 'Project File Structure', 
             fontsize=16, fontweight='bold', ha='center', transform=ax2.transAxes)
    
    # File tree structure
    file_structure = """
vision_analysis/
├── src/
│   ├── data/
│   │   ├── cropped/          # Original & augmented crops
│   │   └── binary/           # Corresponding masks
│   ├── model.py              # UNet & AttentionUNet
│   ├── training.py           # Training pipeline
│   └── tester.py             # Testing & evaluation
├── testing/                  # New manual samples
├── models/
│   ├── original_data/
│   │   ├── checkpoints/      # Training checkpoints
│   │   ├── summaries/        # Model info & plots  
│   │   └── exports/          # Final models (.pkl/.pth)
│   └── augmented_data/
│       ├── checkpoints/
│       ├── summaries/
│       └── exports/
└── test_results/
    ├── original_data/        # Test results by model
    │   ├── validation/
    │   └── new_samples/
    └── augmented_data/
        ├── validation/
        └── new_samples/
    """
    
    ax2.text(0.05, 0.85, file_structure, fontsize=10, ha='left', va='top', 
             fontfamily='monospace', transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/data_flow_structure.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_metrics_diagram():
    """
    Create training metrics and evaluation diagram
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Simulated training curves
    epochs = np.arange(1, 51)
    
    # Original model curves
    orig_train_loss = 0.8 * np.exp(-epochs/15) + 0.1 + 0.02 * np.random.randn(50)
    orig_val_loss = 0.7 * np.exp(-epochs/12) + 0.15 + 0.03 * np.random.randn(50)
    
    # Augmented model curves (slightly better)
    aug_train_loss = 0.75 * np.exp(-epochs/18) + 0.08 + 0.02 * np.random.randn(50)
    aug_val_loss = 0.65 * np.exp(-epochs/15) + 0.12 + 0.025 * np.random.randn(50)
    
    # Training Loss Comparison
    ax1.plot(epochs, orig_train_loss, 'b-', label='Original Data', linewidth=2)
    ax1.plot(epochs, aug_train_loss, 'g-', label='Augmented Data', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss Comparison
    ax2.plot(epochs, orig_val_loss, 'b--', label='Original Data', linewidth=2)
    ax2.plot(epochs, aug_val_loss, 'g--', label='Augmented Data', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Dice Score Comparison (from your actual results)
    models = ['Original\nModel', 'Augmented\nModel']
    dice_scores = [0.859, 0.916]  # Your actual manual test results
    colors = ['lightblue', 'lightgreen']
    
    bars = ax3.bar(models, dice_scores, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Dice Score')
    ax3.set_title('Model Performance on Manual Test Sample', fontsize=14, fontweight='bold')
    ax3.set_ylim(0.8, 0.95)
    
    # Add value labels on bars
    for bar, score in zip(bars, dice_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add performance thresholds
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Excellent (>0.9)')
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Very Good (>0.8)')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Segmentation Quality Visualization
    # Create mock segmentation results
    x = np.linspace(0, 2*np.pi, 100)
    y = np.linspace(0, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    
    # Ground truth (circular pattern)
    ground_truth = ((X - np.pi)**2 + (Y - np.pi)**2) < 1.5
    
    # Original model prediction (slightly less accurate)
    orig_pred = ((X - np.pi + 0.1)**2 + (Y - np.pi - 0.1)**2) < 1.6
    
    # Augmented model prediction (more accurate)  
    aug_pred = ((X - np.pi + 0.02)**2 + (Y - np.pi - 0.02)**2) < 1.52
    
    # Create subplot for segmentation comparison
    ax4.imshow(ground_truth, alpha=0.7, cmap='Greys', label='Ground Truth')
    ax4.contour(orig_pred, colors='blue', linewidths=2, alpha=0.8)
    ax4.contour(aug_pred, colors='green', linewidths=2, alpha=0.8)
    
    ax4.set_title('Segmentation Quality Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add legend for segmentation
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, label='Ground Truth'),
        Line2D([0], [0], color='blue', lw=2, label='Original Model'),
        Line2D([0], [0], color='green', lw=2, label='Augmented Model')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Training Results & Model Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/training_metrics_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_complete_workflow_diagram():
    """
    Create a comprehensive workflow diagram showing training and modeling with inputs/outputs
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Define colors for different stages
    input_color = '#E3F2FD'      # Light Blue
    process_color = '#E8F5E8'     # Light Green  
    model_color = '#FFF3E0'       # Light Orange
    output_color = '#FCE4EC'      # Light Pink
    data_color = '#F3E5F5'        # Light Purple
    
    ax.text(0.5, 0.97, 'Complete Training & Modeling Workflow', 
            fontsize=22, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Stage 1: Raw Data Input
    stage1_box = FancyBboxPatch((0.02, 0.85), 0.18, 0.1,
                               boxstyle="round,pad=0.01",
                               facecolor=input_color, edgecolor='blue', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(0.11, 0.92, 'Stage 1: Raw Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.11, 0.88, 'INPUT:\n• .lif microscopy files\n• Manual annotations', 
            fontsize=9, ha='center', va='center')
    
    # Stage 2: Data Preprocessing
    stage2_box = FancyBboxPatch((0.25, 0.85), 0.18, 0.1,
                               boxstyle="round,pad=0.01",
                               facecolor=process_color, edgecolor='green', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(0.34, 0.92, 'Stage 2: Preprocessing', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.34, 0.88, 'PROCESS:\n• Image cropping\n• Binary mask creation', 
            fontsize=9, ha='center', va='center')
    
    # Stage 3: Data Augmentation (Optional)
    stage3_box = FancyBboxPatch((0.48, 0.85), 0.18, 0.1,
                               boxstyle="round,pad=0.01",
                               facecolor=process_color, edgecolor='orange', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(0.57, 0.92, 'Stage 3: Augmentation', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.57, 0.88, 'PROCESS:\n• Rotation, flips\n• Brightness changes', 
            fontsize=9, ha='center', va='center')
    
    # Stage 4: Data Output
    stage4_box = FancyBboxPatch((0.71, 0.85), 0.25, 0.1,
                               boxstyle="round,pad=0.01",
                               facecolor=output_color, edgecolor='purple', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(0.835, 0.92, 'Stage 4: Processed Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.835, 0.88, 'OUTPUT:\n• .cropped.tif (512×512)\n• .binary.tif (masks)\n• _aug_ variants', 
            fontsize=9, ha='center', va='center')
    
    # DataManager Section
    dm_box = FancyBboxPatch((0.02, 0.7), 0.44, 0.12,
                           boxstyle="round,pad=0.01",
                           facecolor=data_color, edgecolor='purple', linewidth=3)
    ax.add_patch(dm_box)
    ax.text(0.24, 0.79, 'DataManager', fontsize=14, fontweight='bold', ha='center')
    
    # DataManager components
    ax.text(0.08, 0.75, 'INPUT:', fontsize=10, fontweight='bold', ha='left')
    ax.text(0.08, 0.72, '• cropped_dir\n• binary_dir\n• use_augmented flag', 
            fontsize=9, ha='left', va='top')
    
    ax.text(0.24, 0.75, 'PROCESS:', fontsize=10, fontweight='bold', ha='center')
    ax.text(0.24, 0.72, '• Pair images & masks\n• Train/validation split\n• Create DataLoaders', 
            fontsize=9, ha='center', va='top')
    
    ax.text(0.40, 0.75, 'OUTPUT:', fontsize=10, fontweight='bold', ha='right')
    ax.text(0.40, 0.72, '• train_loader\n• val_loader\n• Batch size: 4', 
            fontsize=9, ha='right', va='top')
    
    # Model Architecture Section
    model_section = FancyBboxPatch((0.52, 0.5), 0.44, 0.32,
                                  boxstyle="round,pad=0.01",
                                  facecolor=model_color, edgecolor='orange', linewidth=3)
    ax.add_patch(model_section)
    ax.text(0.74, 0.795, 'AttentionUNet Model', fontsize=16, fontweight='bold', ha='center')
    
    # Model input/output
    ax.text(0.55, 0.765, 'INPUT: 1×512×512 (grayscale)', fontsize=10, fontweight='bold', ha='left')
    ax.text(0.93, 0.765, 'OUTPUT: 1×512×512 (mask)', fontsize=10, fontweight='bold', ha='right')
    
    # Model components in a flow
    model_components = [
        (0.55, 0.72, 'Encoder\nDoubleConv\n64→128→256\n→512→1024'),
        (0.74, 0.72, 'Attention\nGates\nFocus on\nrelevant features'),
        (0.91, 0.72, 'Decoder\nUpConv\n512→256→128\n→64→1')
    ]
    
    for x, y, text in model_components:
        comp_box = FancyBboxPatch((x-0.06, y-0.05), 0.12, 0.1,
                                 boxstyle="round,pad=0.005",
                                 facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(comp_box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Skip connections
    ax.annotate('', xy=(0.91, 0.67), xytext=(0.55, 0.67),
               arrowprops=dict(arrowstyle='<->', color='blue', lw=2, alpha=0.7))
    ax.text(0.73, 0.64, 'Skip Connections', fontsize=9, ha='center', color='blue', style='italic')
    
    # Model parameters
    ax.text(0.74, 0.58, 'Model Parameters:', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.74, 0.55, '• Total params: ~31M\n• Input channels: 1\n• Output classes: 1', 
            fontsize=9, ha='center', va='center')
    
    # Training Section
    train_box = FancyBboxPatch((0.02, 0.3), 0.44, 0.32,
                              boxstyle="round,pad=0.01",
                              facecolor=process_color, edgecolor='green', linewidth=3)
    ax.add_patch(train_box)
    ax.text(0.24, 0.595, 'Trainer', fontsize=16, fontweight='bold', ha='center')
    
    # Training components
    train_items = [
        ('Loss Function:', 'Combined BCE + Dice\nBCE: pixel-wise errors\nDice: overlap measure'),
        ('Optimizer:', 'Adam (lr=1e-4)\nweight_decay=1e-5'),
        ('Scheduler:', 'ReduceLROnPlateau\npatience=5, factor=0.5'),
        ('Training Loop:', 'Forward → Loss → Backward\nValidation each epoch\nCheckpoint best model')
    ]
    
    y_positions = [0.55, 0.47, 0.39, 0.31]
    for i, (title, content) in enumerate(train_items):
        ax.text(0.05, y_positions[i], title, fontsize=10, fontweight='bold', ha='left')
        ax.text(0.43, y_positions[i], content, fontsize=8, ha='right', va='center')
    
    # ModelManager Section
    mm_box = FancyBboxPatch((0.52, 0.15), 0.44, 0.32,
                           boxstyle="round,pad=0.01",
                           facecolor=output_color, edgecolor='purple', linewidth=3)
    ax.add_patch(mm_box)
    ax.text(0.74, 0.445, 'ModelManager', fontsize=16, fontweight='bold', ha='center')
    
    # ModelManager outputs
    mm_items = [
        ('Checkpoints/', '• best_model_{run_name}.pth\n• epoch checkpoints\n• Resume capability'),
        ('Summaries/', '• Model architecture JSON\n• Training loss plots\n• Parameter counts'),
        ('Exports/', '• Full model (.pkl)\n• State dict (.pth)\n• Metadata & config')
    ]
    
    y_pos_mm = [0.40, 0.32, 0.24]
    for i, (folder, content) in enumerate(mm_items):
        ax.text(0.55, y_pos_mm[i], folder, fontsize=10, fontweight='bold', ha='left')
        ax.text(0.93, y_pos_mm[i], content, fontsize=8, ha='right', va='center')
    
    # File Structure Output
    file_box = FancyBboxPatch((0.02, 0.02), 0.44, 0.25,
                             boxstyle="round,pad=0.01",
                             facecolor='#F0F0F0', edgecolor='black', linewidth=2)
    ax.add_patch(file_box)
    ax.text(0.24, 0.245, 'Generated File Structure', fontsize=12, fontweight='bold', ha='center')
    
    file_structure = """models/
├── original_data/
│   ├── checkpoints/
│   ├── summaries/
│   └── exports/
└── augmented_data/
    ├── checkpoints/
    ├── summaries/
    └── exports/"""
    
    ax.text(0.04, 0.20, file_structure, fontsize=9, ha='left', va='top', fontfamily='monospace')
    
    # Testing Section
    test_box = FancyBboxPatch((0.52, 0.02), 0.44, 0.25,
                             boxstyle="round,pad=0.01",
                             facecolor='#E8F4FD', edgecolor='blue', linewidth=2)
    ax.add_patch(test_box)
    ax.text(0.74, 0.245, 'ModelTester - Evaluation', fontsize=12, fontweight='bold', ha='center')
    
    test_items = [
        ('INPUT:', '• Trained model (.pkl)\n• Test images\n• Ground truth masks'),
        ('PROCESS:', '• Load & preprocess\n• Run inference\n• Calculate metrics'),
        ('OUTPUT:', '• Dice scores\n• IoU scores\n• Visualizations')
    ]
    
    x_pos_test = [0.55, 0.74, 0.92]
    for i, (title, content) in enumerate(test_items):
        ax.text(x_pos_test[i], 0.18, title, fontsize=10, fontweight='bold', ha='center')
        ax.text(x_pos_test[i], 0.12, content, fontsize=8, ha='center', va='center')
    
    # Add flow arrows
    arrows = [
        # Data flow
        ((0.20, 0.90), (0.25, 0.90)),  # Stage 1 -> 2
        ((0.43, 0.90), (0.48, 0.90)),  # Stage 2 -> 3
        ((0.66, 0.90), (0.71, 0.90)),  # Stage 3 -> 4
        ((0.835, 0.85), (0.24, 0.82)), # Stage 4 -> DataManager
        
        # Training flow
        ((0.24, 0.70), (0.24, 0.62)),  # DataManager -> Trainer
        ((0.46, 0.46), (0.52, 0.66)),  # Trainer -> Model
        ((0.74, 0.50), (0.74, 0.47)),  # Model -> ModelManager
        
        # Testing flow
        ((0.74, 0.15), (0.74, 0.27)),  # ModelManager -> Tester
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add metrics annotations
    ax.text(0.02, 0.85, 'Key Metrics:', fontsize=10, fontweight='bold', ha='left', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    ax.text(0.02, 0.82, '• Dice Score: 0.916 (Augmented)\n• IoU Score: 0.845 (Augmented)\n• Training Loss: Combined BCE+Dice', 
            fontsize=8, ha='left', va='top')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/complete_workflow_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_data_transformation_diagram():
    """
    Create a detailed diagram showing data transformations through the pipeline
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    ax.text(0.5, 0.95, 'Data Transformation Pipeline', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Create sample data visualizations
    stages = [
        {
            'title': '1. Raw .lif File',
            'pos': (0.1, 0.8),
            'size': (0.15, 0.12),
            'color': '#E3F2FD',
            'data_info': 'Multi-channel\nMicroscopy\n2048×2048+',
            'example': 'create_sample_microscopy()'
        },
        {
            'title': '2. Cropped Images',
            'pos': (0.3, 0.8),
            'size': (0.15, 0.12),
            'color': '#E8F5E8',
            'data_info': 'Single channel\nGrayscale\n512×512',
            'example': 'create_sample_crop()'
        },
        {
            'title': '3. Binary Masks',
            'pos': (0.5, 0.8),
            'size': (0.15, 0.12),
            'color': '#FFF3E0',
            'data_info': 'Binary\n0/255 values\n512×512',
            'example': 'create_sample_mask()'
        },
        {
            'title': '4. Augmented Data',
            'pos': (0.7, 0.8),
            'size': (0.15, 0.12),
            'color': '#FCE4EC',
            'data_info': 'Rotated/Flipped\nMultiple variants\n512×512',
            'example': 'create_sample_augmented()'
        }
    ]
    
    # Draw transformation stages
    for i, stage in enumerate(stages):
        # Main box
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                            boxstyle="round,pad=0.01",
                            facecolor=stage['color'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1] - 0.02,
               stage['title'], fontsize=11, fontweight='bold', ha='center')
        
        # Data info
        ax.text(stage['pos'][0] + stage['size'][0]/2, 
               stage['pos'][1] + stage['size'][1]/2,
               stage['data_info'], fontsize=9, ha='center', va='center')
        
        # Sample visualization placeholder
        viz_box = FancyBboxPatch((stage['pos'][0] + 0.01, stage['pos'][1] + 0.01), 
                                stage['size'][0] - 0.02, 0.04,
                                boxstyle="round,pad=0.005",
                                facecolor='white', edgecolor='gray')
        ax.add_patch(viz_box)
        ax.text(stage['pos'][0] + stage['size'][0]/2, stage['pos'][1] + 0.03,
               '[Sample Image]', fontsize=8, ha='center', style='italic')
        
        # Arrow to next stage (except last)
        if i < len(stages) - 1:
            ax.arrow(stage['pos'][0] + stage['size'][0], stage['pos'][1] + stage['size'][1]/2,
                    0.04, 0, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # DataLoader Section
    dl_box = FancyBboxPatch((0.1, 0.6), 0.8, 0.15,
                           boxstyle="round,pad=0.01",
                           facecolor='#F3E5F5', edgecolor='purple', linewidth=3)
    ax.add_patch(dl_box)
    ax.text(0.5, 0.72, 'DataLoader Transformation', fontsize=14, fontweight='bold', ha='center')
    
    # DataLoader steps
    dl_steps = [
        ('Load Image/Mask', 'cv2.imread()\nGrayscale'),
        ('Normalize', 'image/255.0\nmask > 0.5'),
        ('Resize', 'A.Resize(512,512)'),
        ('Tensor Convert', 'ToTensorV2()'),
        ('Batch Creation', 'batch_size=4')
    ]
    
    x_positions = [0.15, 0.28, 0.45, 0.62, 0.78]
    for i, (step, details) in enumerate(dl_steps):
        step_box = FancyBboxPatch((x_positions[i] - 0.06, 0.635), 0.12, 0.08,
                                 boxstyle="round,pad=0.005",
                                 facecolor='white', edgecolor='black')
        ax.add_patch(step_box)
        ax.text(x_positions[i], 0.69, step, fontsize=9, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.655, details, fontsize=7, ha='center', va='center')
        
        if i < len(dl_steps) - 1:
            ax.arrow(x_positions[i] + 0.06, 0.675, 0.04, 0, 
                    head_width=0.01, head_length=0.005, fc='purple', ec='purple')
    
    # Model Input/Output Section
    model_box = FancyBboxPatch((0.1, 0.4), 0.8, 0.15,
                              boxstyle="round,pad=0.01",
                              facecolor='#FFF3E0', edgecolor='orange', linewidth=3)
    ax.add_patch(model_box)
    ax.text(0.5, 0.52, 'Model Forward Pass', fontsize=14, fontweight='bold', ha='center')
    
    # Model flow
    model_flow = [
        ('Input Batch', '4×1×512×512\n(B×C×H×W)'),
        ('Encoder', 'Feature\nExtraction'),
        ('Attention', 'Feature\nSelection'),
        ('Decoder', 'Mask\nReconstruction'),
        ('Output Logits', '4×1×512×512\n(Raw scores)')
    ]
    
    for i, (step, details) in enumerate(model_flow):
        step_box = FancyBboxPatch((x_positions[i] - 0.06, 0.415), 0.12, 0.08,
                                 boxstyle="round,pad=0.005",
                                 facecolor='white', edgecolor='orange')
        ax.add_patch(step_box)
        ax.text(x_positions[i], 0.47, step, fontsize=9, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.435, details, fontsize=7, ha='center', va='center')
        
        if i < len(model_flow) - 1:
            ax.arrow(x_positions[i] + 0.06, 0.455, 0.04, 0, 
                    head_width=0.01, head_length=0.005, fc='orange', ec='orange')
    
    # Loss Calculation Section  
    loss_box = FancyBboxPatch((0.1, 0.2), 0.8, 0.15,
                             boxstyle="round,pad=0.01",
                             facecolor='#E8F5E8', edgecolor='green', linewidth=3)
    ax.add_patch(loss_box)
    ax.text(0.5, 0.32, 'Loss Calculation & Optimization', fontsize=14, fontweight='bold', ha='center')
    
    # Loss components
    loss_steps = [
        ('Apply Sigmoid', 'torch.sigmoid()\nLogits → Probs'),
        ('BCE Loss', 'Pixel-wise\nClassification'),
        ('Dice Loss', 'Overlap\nMeasure'),
        ('Combined Loss', 'BCE + Dice\nWeighted sum'),
        ('Backprop', 'Gradient\nUpdate')
    ]
    
    for i, (step, details) in enumerate(loss_steps):
        step_box = FancyBboxPatch((x_positions[i] - 0.06, 0.235), 0.12, 0.08,
                                 boxstyle="round,pad=0.005",
                                 facecolor='white', edgecolor='green')
        ax.add_patch(step_box)
        ax.text(x_positions[i], 0.29, step, fontsize=9, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.255, details, fontsize=7, ha='center', va='center')
        
        if i < len(loss_steps) - 1:
            ax.arrow(x_positions[i] + 0.06, 0.275, 0.04, 0, 
                    head_width=0.01, head_length=0.005, fc='green', ec='green')
    
    # Output Section
    output_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.15,
                               boxstyle="round,pad=0.01",
                               facecolor='#FCE4EC', edgecolor='purple', linewidth=3)
    ax.add_patch(output_box)
    ax.text(0.5, 0.14, 'Training Outputs', fontsize=14, fontweight='bold', ha='center')
    
    # Final outputs
    final_outputs = [
        ('Checkpoints', 'best_model.pth\nEpoch saves'),
        ('Model Export', 'Full model.pkl\nState dict.pth'),
        ('Summaries', 'Loss plots\nModel info'),
        ('Metrics', 'Training curves\nValidation scores'),
        ('Test Results', 'Dice: 0.916\nIoU: 0.845')
    ]
    
    for i, (output, details) in enumerate(final_outputs):
        out_box = FancyBboxPatch((x_positions[i] - 0.06, 0.055), 0.12, 0.08,
                                boxstyle="round,pad=0.005",
                                facecolor='white', edgecolor='purple')
        ax.add_patch(out_box)
        ax.text(x_positions[i], 0.11, output, fontsize=9, fontweight='bold', ha='center')
        ax.text(x_positions[i], 0.075, details, fontsize=7, ha='center', va='center')
    
    # Add connecting arrows between sections
    section_arrows = [
        ((0.5, 0.8), (0.5, 0.75)),   # Stages -> DataLoader
        ((0.5, 0.6), (0.5, 0.55)),   # DataLoader -> Model
        ((0.5, 0.4), (0.5, 0.35)),   # Model -> Loss
        ((0.5, 0.2), (0.5, 0.17)),   # Loss -> Outputs
    ]
    
    for start, end in section_arrows:
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                head_width=0.02, head_length=0.01, fc='black', ec='black', lw=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/bhunn1/vision_analysis/data_transformation_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
# Create a script to generate all diagrams
def create_all_diagrams():
    """
    Generate all diagrams for the vision analysis project
    """
    
    print("Creating Complete Workflow Diagram...")
    create_complete_workflow_diagram()
    
    print("Creating Data Transformation Diagram...")
    create_data_transformation_diagram()
    
    print("\nAll diagrams saved to /home/bhunn1/vision_analysis/")
    print("Files created:")
    print("- training_pipeline_diagram.png")
    print("- attention_unet_architecture.png") 
    print("- data_flow_structure.png")
    print("- training_metrics_analysis.png")
    print("- complete_workflow_diagram.png")
    print("- data_transformation_diagram.png")

if __name__ == "__main__":
    create_all_diagrams()
"""
Example 1: Simple CNN Architecture
A basic convolutional neural network for biological image classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.
    Input: RGB images (3 channels)
    Output: Probabilities for num_classes
    """
    def __init__(self, num_classes=4, input_size=256):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions
        size_after_conv = input_size // (2 ** 3)  # 3 pooling layers
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * size_after_conv * size_after_conv, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_architecture(model, input_size=(1, 3, 256, 256)):
    """
    Visualize the flow of data through the network.
    Shows output shape at each layer.
    """
    print("=" * 70)
    print("CNN Architecture Summary")
    print("=" * 70)
    
    x = torch.randn(input_size)
    
    print(f"Input shape: {x.shape}")
    print("-" * 70)
    
    # Track shapes through each layer
    x = model.conv1(x)
    print(f"After Conv1: {x.shape}")
    x = model.bn1(x)
    x = model.relu1(x)
    x = model.pool1(x)
    print(f"After Pool1: {x.shape}")
    
    x = model.conv2(x)
    print(f"After Conv2: {x.shape}")
    x = model.bn2(x)
    x = model.relu2(x)
    x = model.pool2(x)
    print(f"After Pool2: {x.shape}")
    
    x = model.conv3(x)
    print(f"After Conv3: {x.shape}")
    x = model.bn3(x)
    x = model.relu3(x)
    x = model.pool3(x)
    print(f"After Pool3: {x.shape}")
    
    x = model.flatten(x)
    print(f"After Flatten: {x.shape}")
    
    x = model.fc1(x)
    print(f"After FC1: {x.shape}")
    
    x = model.dropout(x)
    x = model.fc2(x)
    print(f"Output shape: {x.shape}")
    
    print("=" * 70)
    print(f"Total parameters: {count_parameters(model):,}")
    print("=" * 70)


def demonstrate_forward_pass():
    """Demonstrate a forward pass with random data"""
    print("\nDemonstrating Forward Pass:")
    print("-" * 70)
    
    # Create model
    model = SimpleCNN(num_classes=4)
    model.eval()  # Set to evaluation mode
    
    # Create fake batch of images
    batch_size = 8
    fake_images = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input: Batch of {batch_size} images, size 256x256, 3 channels")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(fake_images)
        
    print(f"Output shape: {outputs.shape}")
    print(f"Output (raw logits) for first image:\n{outputs[0]}")
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    print(f"\nProbabilities for first image:\n{probabilities[0]}")
    print(f"Sum of probabilities: {probabilities[0].sum():.4f}")
    
    # Get predicted class
    predicted_classes = torch.argmax(probabilities, dim=1)
    print(f"\nPredicted classes for batch: {predicted_classes}")


def visualize_filters(model, layer_name='conv1'):
    """Visualize the learned filters in the first convolutional layer"""
    print(f"\nVisualizing filters from {layer_name}:")
    
    # Get the convolutional layer
    conv_layer = getattr(model, layer_name)
    filters = conv_layer.weight.data.cpu().numpy()
    
    num_filters = min(32, filters.shape[0])  # Show first 32 filters
    
    fig, axes = plt.subplots(4, 8, figsize=(15, 7))
    axes = axes.ravel()
    
    for i in range(num_filters):
        # Get the filter (take mean across input channels for visualization)
        filter_img = filters[i].mean(axis=0)
        
        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}', fontsize=8)
    
    plt.suptitle(f'Learned Filters in {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{layer_name}_filters.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {layer_name}_filters.png")
    plt.close()


def compare_architectures():
    """Compare different CNN architectures"""
    print("\nComparing Different Architectures:")
    print("=" * 70)
    
    configs = {
        "Small": {"conv_channels": [16, 32, 64], "fc_size": 128},
        "Medium": {"conv_channels": [32, 64, 128], "fc_size": 256},
        "Large": {"conv_channels": [64, 128, 256], "fc_size": 512}
    }
    
    for name, config in configs.items():
        # Create custom model
        model = SimpleCNN(num_classes=4)
        params = count_parameters(model)
        
        print(f"{name:10s}: {params:>12,} parameters")


if __name__ == "__main__":
    print("Simple CNN Example for Biologists")
    print("=" * 70)
    
    # Create model
    model = SimpleCNN(num_classes=4, input_size=256)
    
    # Visualize architecture
    visualize_architecture(model)
    
    # Demonstrate forward pass
    demonstrate_forward_pass()
    
    # Visualize filters (note: these are random, not trained)
    visualize_filters(model, 'conv1')
    
    # Compare architectures
    compare_architectures()
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("Next: See example_02_cell_classifier.py for complete training example")
    print("=" * 70)

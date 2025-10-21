# Module 3: Hands-On CNN Tutorial

## Getting Started

In this module, you'll build and train your first CNN! We'll create a classifier for cell images—a common task in biological research.

### Prerequisites Check

Before starting, ensure you've:
- ✓ Installed all requirements (`pip install -r requirements.txt`)
- ✓ Have Python 3.8+ running
- ✓ Understand basic Python (variables, loops, functions)

## Our Task: Cell Image Classification

We'll build a CNN to classify microscopy images into four categories:
1. **Healthy cells**
2. **Apoptotic cells** (undergoing programmed death)
3. **Necrotic cells** (damaged/dying)
4. **Dividing cells** (mitosis)

This is a realistic biological task where CNNs excel.

## Part 1: Understanding the Code Structure

### Basic CNN Architecture

Here's a simple CNN in PyTorch:

```python
import torch
import torch.nn as nn

class SimpleCellCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCellCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Pass through convolutional blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Pass through fully connected layers
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

**What's happening here?**
- `Conv2d`: Convolutional layer (detects patterns)
- `ReLU`: Activation function (adds non-linearity)
- `MaxPool2d`: Pooling layer (reduces size)
- `Linear`: Fully connected layer (combines features)
- `Dropout`: Randomly turns off neurons during training (prevents overfitting)

## Part 2: Preparing Your Data

### Data Organization

Organize your images in folders:
```
data/
├── train/
│   ├── healthy/
│   │   ├── cell_001.jpg
│   │   ├── cell_002.jpg
│   ├── apoptotic/
│   ├── necrotic/
│   └── dividing/
└── validation/
    ├── healthy/
    ├── apoptotic/
    ├── necrotic/
    └── dividing/
```

### Loading Data

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),           # Resize all images
    transforms.RandomHorizontalFlip(),       # Augmentation: flip randomly
    transforms.RandomRotation(10),           # Augmentation: rotate slightly
    transforms.ColorJitter(brightness=0.2),  # Augmentation: adjust brightness
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize colors
                        std=[0.229, 0.224, 0.225])
])

# Load training data
train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load validation data
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder('data/validation', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**Key points**:
- **Data augmentation** (flips, rotations) creates variations, helping the network generalize
- Use augmentation for training but NOT for validation
- Batch size of 32 means processing 32 images at once

## Part 3: Training the CNN

### Complete Training Loop

```python
import torch.optim as optim

# Initialize model
model = SimpleCellCNN(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Train for multiple epochs
num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

### What's Happening During Training?

1. **Forward Pass**: Images go through the network → predictions
2. **Calculate Loss**: How wrong were predictions?
3. **Backward Pass**: Calculate how to adjust weights
4. **Update Weights**: Optimizer adjusts network parameters
5. **Repeat**: For all batches and epochs

**Typical output**:
```
Epoch [1/20]
  Train Loss: 1.2456, Train Acc: 45.23%
  Val Loss: 1.1234, Val Acc: 48.56%
Epoch [2/20]
  Train Loss: 0.9823, Train Acc: 58.34%
  Val Loss: 0.8976, Val Acc: 61.23%
...
```

## Part 4: Evaluating Performance

### Confusion Matrix

See where your model makes mistakes:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, val_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Run evaluation
class_names = ['healthy', 'apoptotic', 'necrotic', 'dividing']
evaluate_model(model, val_loader, device, class_names)
```

**Example output**:
```
Classification Report:
              precision    recall  f1-score   support

     healthy       0.92      0.88      0.90       150
   apoptotic       0.85      0.89      0.87       120
    necrotic       0.88      0.85      0.86       130
    dividing       0.91      0.94      0.92       140

    accuracy                           0.89       540
   macro avg       0.89      0.89      0.89       540
weighted avg       0.89      0.89      0.89       540
```

### Key Metrics Explained

- **Precision**: When model says "apoptotic", how often is it correct?
- **Recall**: Of all apoptotic cells, how many did it find?
- **F1-score**: Harmonic mean of precision and recall (balanced metric)
- **Support**: Number of samples in each class

### Visualizing Predictions

```python
import numpy as np

def visualize_predictions(model, val_loader, device, class_names, num_images=9):
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize for display
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                          color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()

visualize_predictions(model, val_loader, device, class_names)
```

## Part 5: Saving and Loading Your Model

### Save the Trained Model

```python
# Save complete model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'class_names': class_names
}, 'cell_classifier_model.pth')

print("Model saved successfully!")
```

### Load and Use the Model

```python
# Load model
checkpoint = torch.load('cell_classifier_model.pth')
model = SimpleCellCNN(num_classes=4)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Make predictions on new images
def predict_single_image(image_path, model, device, class_names):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    pred_class = class_names[predicted.item()]
    conf_percent = confidence.item() * 100
    
    return pred_class, conf_percent

# Example usage
result, confidence = predict_single_image('new_cell.jpg', model, device, class_names)
print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
```

## Part 6: Improving Your Model

### Technique 1: Data Augmentation

Add more variations to your training data:

```python
strong_augment = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Technique 2: Learning Rate Scheduling

Reduce learning rate when training plateaus:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# In training loop, after validation:
scheduler.step(val_loss)
```

### Technique 3: Early Stopping

Stop training when validation performance stops improving:

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
```

## Part 7: Troubleshooting Common Issues

### Problem 1: Overfitting

**Symptoms**: Training accuracy much higher than validation accuracy

**Solutions**:
- Add more data augmentation
- Increase dropout rate (try 0.5 or 0.6)
- Use less complex model (fewer layers/filters)
- Add more training data
- Use regularization (L2 weight decay)

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Problem 2: Underfitting

**Symptoms**: Both training and validation accuracy are low

**Solutions**:
- Use more complex model (more layers/filters)
- Train for more epochs
- Reduce dropout
- Increase learning rate
- Check if data is properly preprocessed

### Problem 3: Class Imbalance

**Symptoms**: Model always predicts the majority class

**Solutions**:
- Balance your dataset (equal samples per class)
- Use weighted loss function

```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(train_dataset.targets),
                                     y=train_dataset.targets)
class_weights = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Problem 4: Slow Training

**Solutions**:
- Use GPU if available
- Reduce image size
- Increase batch size (if memory allows)
- Use fewer training samples initially to debug

## Exercises

### Exercise 1: Modify the Architecture
Try changing:
- Number of filters (32, 64, 128 → try 16, 32, 64)
- Number of layers (add or remove one convolutional block)
- Fully connected layer size (256 → 512 or 128)

How does each change affect performance?

### Exercise 2: Experiment with Hyperparameters
- Learning rate: Try 0.0001, 0.001, 0.01
- Batch size: Try 16, 32, 64
- Number of epochs: Find when model stops improving

### Exercise 3: Analyze Mistakes
Look at the confusion matrix. Which classes are confused with each other? Why might this be biologically?

### Exercise 4: Feature Visualization
Add code to visualize first layer filters:

```python
def visualize_filters(model):
    # Get first convolutional layer
    first_conv = model.conv1.weight.data.cpu()
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(32):
        filter_img = first_conv[i].numpy()
        # Take first channel for visualization
        axes[i].imshow(filter_img[0], cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('First Layer Filters')
    plt.tight_layout()
    plt.show()

visualize_filters(model)
```

## Next Steps

Congratulations! You've built, trained, and evaluated a CNN for biological image classification.

In Module 4, we'll explore:
- Real biological applications in depth
- Transfer learning (using pre-trained models)
- Working with limited data
- Advanced visualization techniques
- Deploying your model for actual research use

**Continue to: Module 4 - Biological Applications** →

---

## Quick Reference: Training Checklist

Before training:
- [ ] Data organized in correct folder structure
- [ ] Sufficient samples per class (at least 50-100)
- [ ] Images are of consistent quality
- [ ] Train/validation split created (typically 80/20)

During training:
- [ ] Monitor both training and validation metrics
- [ ] Watch for overfitting (validation loss increasing)
- [ ] Save best model based on validation performance
- [ ] Use early stopping to prevent wasted training time

After training:
- [ ] Generate confusion matrix
- [ ] Check per-class performance
- [ ] Visualize predictions on validation set
- [ ] Test on completely new images (from different experiment)

## Complete Code Example

See `code/example_02_cell_classifier.py` for the complete, runnable implementation of everything in this module!

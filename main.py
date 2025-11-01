# ============================================================
# Complete ResNet Training on EuroSAT Dataset
# Compatible with Google Colab
# ============================================================

# Install required packages
!pip install torch torchvision pillow matplotlib scikit-learn tqdm

# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================
# 1. DATASET SETUP AND DOWNLOAD
# ============================================================

# Download EuroSAT dataset
!wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
!unzip -q EuroSAT.zip

# Configuration parameters
DATASET_PATH = './2750'  # Path to extracted EuroSAT dataset
BATCH_SIZE = 32
IMAGE_SIZE = 64  # EuroSAT images are 64x64
NUM_CLASSES = 10  # EuroSAT has 10 land use classes
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_WORKERS = 2
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. DATA TRANSFORMS AND AUGMENTATION
# ============================================================

# Data transforms for training (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Data transforms for validation and testing (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# 3. LOAD DATASET AND CREATE DATA LOADERS
# ============================================================

# Load the full dataset
full_dataset = datasets.ImageFolder(DATASET_PATH)

# Get class names
class_names = full_dataset.classes
print(f"Classes: {class_names}")
print(f"Number of classes: {len(class_names)}")
print(f"Total images: {len(full_dataset)}")

# Split dataset into train, validation, and test sets
dataset_size = len(full_dataset)
test_size = int(TEST_SPLIT * dataset_size)
val_size = int(VALIDATION_SPLIT * (dataset_size - test_size))
train_size = dataset_size - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply transforms
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    pin_memory=USE_CUDA
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=USE_CUDA
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=USE_CUDA
)

print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")
print(f"Test set: {len(test_dataset)} images")

# ============================================================
# 4. VISUALIZE SAMPLE IMAGES
# ============================================================

def imshow(inp, title=None):
    """Display image from tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[:8])
imshow(out, title=[class_names[x] for x in classes[:8]])
plt.show()

# ============================================================
# 5. CREATE RESNET MODEL
# ============================================================

def create_resnet_model(num_classes, pretrained=True, model_type='resnet50'):
    """
    Create ResNet model for EuroSAT classification.
    
    Args:
        num_classes: Number of output classes (10 for EuroSAT)
        pretrained: Whether to use ImageNet pretrained weights
        model_type: Type of ResNet ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
    """
    # Load pretrained ResNet model
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Create model (using ResNet-50)
model = create_resnet_model(NUM_CLASSES, pretrained=True, model_type='resnet50')
model = model.to(device)

print(f"\nModel architecture:\n{model}")

# ============================================================
# 6. DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer - using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler - reduces learning rate when validation loss plateaus
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

# Alternative: Step learning rate scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """
    Train the ResNet model on EuroSAT dataset.
    """
    since = time.time()
    
    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
                dataset_size = len(val_dataset)
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data with progress bar
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()} Phase')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Step the scheduler
                scheduler.step(epoch_loss)
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'New best model! Validation Accuracy: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

model, history = train_model(
    model, 
    criterion, 
    optimizer, 
    scheduler, 
    num_epochs=NUM_EPOCHS
)

# Save the trained model
torch.save(model.state_dict(), 'resnet_eurosat_best.pt')
print("\nModel saved as 'resnet_eurosat_best.pt'")

# ============================================================
# 9. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot learning rate
    axes[2].plot(history['learning_rates'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# ============================================================
# 10. EVALUATE ON TEST SET
# ============================================================

def evaluate_model(model, test_loader, class_names):
    """Evaluate model on test set and generate metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_dataset)
    test_acc = 100 * correct / total
    
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_acc, all_preds, all_labels

# Evaluate the model
test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, class_names)

# ============================================================
# 11. VISUALIZE PREDICTIONS
# ============================================================

def visualize_predictions(model, test_loader, class_names, num_images=16):
    """Visualize model predictions on test images."""
    model.eval()
    
    images_so_far = 0
    fig = plt.figure(figsize=(20, 20))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(4, 4, images_so_far)
                ax.axis('off')
                
                # Get prediction and true label
                pred_label = class_names[preds[j]]
                true_label = class_names[labels[j]]
                
                # Set title color based on correctness
                color = 'green' if pred_label == true_label else 'red'
                ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', 
                           color=color, fontsize=10)
                
                # Display image
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    return

visualize_predictions(model, test_loader, class_names, num_images=16)

# ============================================================
# 12. INFERENCE ON NEW IMAGES
# ============================================================

def predict_single_image(model, image_path, class_names):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the image file
        class_names: List of class names
    
    Returns:
        Predicted class name and confidence score
    """
    from PIL import Image
    
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = val_test_transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Display image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence_score:.2f}%', 
              fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence_score

# Example usage (uncomment to use):
# predicted_class, confidence = predict_single_image(model, 'path/to/your/image.jpg', class_names)
# print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best Validation Accuracy: {max(history['val_acc'])*100:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Model saved as: resnet_eurosat_best.pt")
print("="*60)

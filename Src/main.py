import torch
from preprocess import get_data_loaders
from model import SimpleCNN
from train import train_model
from evaluate import evaluate_model
from visualize import plot_metrics

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 20
patience = 5
image_size = 224

# Dataset paths
train_dir = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_dir = '/kaggle/input/brain-tumor-mri-dataset/Testing'

# Get DataLoaders and number of classes
train_loader, val_loader, test_loader, num_classes = get_data_loaders(train_dir, test_dir, batch_size, image_size)

# Initialize Model
model = SimpleCNN(num_classes).to(device)

# Train Model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, num_epochs, learning_rate, patience, device
)

# Load Best Model and Evaluate
model.load_state_dict(torch.load("best_model.pth"))
evaluate_model(model, test_loader, test_loader.dataset, device)

# Visualize Results
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# %% [markdown]
# # Cat vs Not-Cat Transfer Learning Project
#
# Run this file as a notebook-style script in Kaggle, Colab, Jupyter, or VS Code.
# Update only `CAT_SOURCE` and `NOT_CAT_SOURCE` for your cloud environment.

# %% Case 1: Imports and basic settings
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42
MAX_IMAGES_PER_CLASS = 1500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current compute device: {device}")
if device.type == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")


# %% Case 2: Dataset paths
# Change these two paths for your cloud server.
CAT_SOURCE = "/kaggle/input/datasets/crawford/cat-dataset"
NOT_CAT_SOURCE = "/kaggle/input/your-not-cat-dataset-path"

CLASS_NAMES = ["Cat", "Non-Cat"]


def collect_image_paths(source_dir, max_images=None):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source = Path(source_dir)

    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source_dir}")

    paths = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    paths = sorted(paths)

    if max_images is not None:
        paths = paths[:max_images]

    return paths


cat_paths = collect_image_paths(CAT_SOURCE, MAX_IMAGES_PER_CLASS)
not_cat_paths = collect_image_paths(NOT_CAT_SOURCE, MAX_IMAGES_PER_CLASS)

print("Cat images:", len(cat_paths))
print("Not cat images:", len(not_cat_paths))

if len(cat_paths) == 0:
    raise ValueError("No cat images found. Check CAT_SOURCE.")

if len(not_cat_paths) == 0:
    raise ValueError("No not-cat images found. Check NOT_CAT_SOURCE.")


# %% Case 3: PyTorch Dataset and DataLoader
class ImagePathDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

samples = [(path, 0) for path in cat_paths] + [(path, 1) for path in not_cat_paths]

random.seed(RANDOM_SEED)
random.shuffle(samples)

full_dataset = ImagePathDataset(samples, transform=transform)

dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

indices = list(range(dataset_size))
train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# %% Case 4: Preview training images
def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor.cpu() * std + mean
    return image_tensor.clamp(0, 1)


images, labels = next(iter(train_loader))

plt.figure(figsize=(10, 10))
for i in range(min(9, len(images))):
    image = denormalize(images[i]).permute(1, 2, 0).numpy()

    plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[labels[i].item()])
    plt.axis("off")

plt.show()


# %% Case 5: Build transfer learning model
print("Loading ResNet50 model...")

try:
    transfer_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    print("Loaded ImageNet pre-trained weights.")
except Exception as error:
    print("Could not load pre-trained weights.")
    print("Falling back to weights=None. If you are on Kaggle, turn Internet on for better results.")
    print(f"Details: {error}")
    transfer_model = models.resnet50(weights=None)

for param in transfer_model.parameters():
    param.requires_grad = False

in_features = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(in_features, 2)
transfer_model = transfer_model.to(device)

trainable_params = [name for name, p in transfer_model.named_parameters() if p.requires_grad]

print("Model built successfully. Trainable layers:")
for name in trainable_params:
    print(f"  - {name}")


# %% Case 6: Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda param: param.requires_grad, transfer_model.parameters()),
    lr=LEARNING_RATE,
)

print("Optimizer setup complete.")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")


# %% Case 7: Training and validation
best_model_wts = transfer_model.state_dict()
best_val_acc = 0.0

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    transfer_model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = transfer_model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        total_train_samples += inputs.size(0)

    epoch_train_loss = running_loss / total_train_samples
    epoch_train_acc = running_corrects.double() / total_train_samples

    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc.item())

    transfer_model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_val_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = transfer_model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels)
            total_val_samples += inputs.size(0)

    epoch_val_loss = val_loss / total_val_samples
    epoch_val_acc = val_corrects.double() / total_val_samples

    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc.item())

    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_wts = {key: value.cpu().clone() for key, value in transfer_model.state_dict().items()}
        print("New best model saved.")

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
transfer_model.load_state_dict(best_model_wts)
transfer_model = transfer_model.to(device)


# %% Case 8: Final evaluation and confusion matrix
print("Evaluating the best model on the test set...")

all_preds = []
all_labels = []

transfer_model.eval()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = transfer_model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n================ EVALUATION REPORT ================")
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
print(report)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.title("Transfer Learning Confusion Matrix")
plt.show()


# %% Case 9: Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()


# %% Case 10: Save model
MODEL_PATH = "cat_not_cat_resnet50.pth"

torch.save(
    {
        "model_state_dict": transfer_model.state_dict(),
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
    },
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")

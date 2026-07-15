# %% [markdown]
# # Cat vs Not-Cat: Kaggle Transfer Learning (ResNet50)
#
# Before running:
# 1. In Kaggle, click **Add Input** and add the dataset containing your images.
# 2. Set the two paths in Case 2 to the correct folders.
# 3. Enable GPU: Settings -> Accelerator -> GPU T4 x2 (or GPU P100).
# 4. Enable Internet if you want ImageNet pretrained weights to download.
#
# The folder layout can be any depth. For example:
# /kaggle/input/my-images/cat/...
# /kaggle/input/my-images/not_cat/...

# %% Case 1: Imports and settings
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
MAX_IMAGES_PER_CLASS = 1500  # Set to None to use every image.
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)


# %% Case 2: Set your Kaggle input paths
# Change ONLY these two lines after adding your Kaggle dataset as an input.
# Example: CAT_SOURCE = "/kaggle/input/cat-not-cat-dataset/cat"
#          NOT_CAT_SOURCE = "/kaggle/input/cat-not-cat-dataset/not_cat"
CAT_SOURCE = "/kaggle/input/REPLACE_WITH_YOUR_DATASET_FOLDER/cat"
NOT_CAT_SOURCE = "/kaggle/input/REPLACE_WITH_YOUR_DATASET_FOLDER/not_cat"

CLASS_NAMES = ["Cat", "Non-Cat"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_valid_images(source_dir, max_images=None):
    """Find readable image files recursively without copying any files."""
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(
            f"Folder not found: {source}\n"
            "Check the exact folder name under /kaggle/input in the left file panel."
        )

    paths = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    valid_paths = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            valid_paths.append(path)
        except (UnidentifiedImageError, OSError, ValueError):
            pass

        if max_images is not None and len(valid_paths) >= max_images:
            break

    return valid_paths


cat_paths = collect_valid_images(CAT_SOURCE, MAX_IMAGES_PER_CLASS)
not_cat_paths = collect_valid_images(NOT_CAT_SOURCE, MAX_IMAGES_PER_CLASS)

print(f"Cat images: {len(cat_paths)}")
print(f"Non-cat images: {len(not_cat_paths)}")

if not cat_paths or not not_cat_paths:
    raise ValueError(
        "Both folders must contain at least one readable image. "
        "A binary classifier needs cat images and non-cat images."
    )


# %% Case 3: Split data and build PyTorch DataLoaders
all_paths = cat_paths + not_cat_paths
all_labels = [0] * len(cat_paths) + [1] * len(not_cat_paths)

# Stratification keeps the Cat / Non-Cat ratio similar in every split.
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=all_labels,
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[index]


train_dataset = ImagePathDataset(train_paths, train_labels, train_transform)
val_dataset = ImagePathDataset(val_paths, val_labels, eval_transform)
test_dataset = ImagePathDataset(test_paths, test_labels, eval_transform)

loader_options = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": device.type == "cuda",
}
train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")


# %% Case 4: Preview training images
def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor.cpu() * std + mean).clamp(0, 1)


images, labels = next(iter(train_loader))
plt.figure(figsize=(9, 9))
for index in range(min(9, len(images))):
    plt.subplot(3, 3, index + 1)
    plt.imshow(denormalize(images[index]).permute(1, 2, 0))
    plt.title(CLASS_NAMES[labels[index].item()])
    plt.axis("off")
plt.tight_layout()
plt.show()


# %% Case 5: Build ResNet50 transfer-learning model
print("Loading ResNet50...")
try:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    print("Using ImageNet pretrained weights.")
except Exception as error:
    print(f"Pretrained weights unavailable: {error}")
    print("Continuing with randomly initialized weights. Enable Internet for better results.")
    model = models.resnet50(weights=None)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model = model.to(device)

print("Trainable layer: fc")


# %% Case 6: Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# %% Case 7: Train and validate
def run_epoch(data_loader, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = -1.0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
        f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print("Best validation model updated.")

model.load_state_dict(best_model_weights)
print(f"Best validation accuracy: {best_val_acc:.4f}")


# %% Case 8: Test-set evaluation
model.eval()
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device, non_blocking=True))
        all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        all_true_labels.extend(labels.tolist())

print("\nTest report")
print(classification_report(all_true_labels, all_predictions, target_names=CLASS_NAMES, digits=4))

matrix = confusion_matrix(all_true_labels, all_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()


# %% Case 9: Training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], marker="o", label="Train")
plt.plot(history["val_loss"], marker="o", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], marker="o", label="Train")
plt.plot(history["val_acc"], marker="o", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves.png", dpi=160)
plt.show()


# %% Case 10: Save model to Kaggle output
MODEL_PATH = "/kaggle/working/cat_not_cat_resnet50.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMG_SIZE,
        "best_validation_accuracy": best_val_acc,
    },
    MODEL_PATH,
)

print(f"Saved model: {MODEL_PATH}")
print("Saved curve image: /kaggle/working/training_curves.png")

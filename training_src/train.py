import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2

# Import the project dataset and models
from dataset import CatNonCatDataset, get_transforms
from baseline_model import get_baseline_model
from transfer_model import get_transfer_model

def train_model(args):
    # 1. Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    if device.type == "cpu":
        print("[WARNING] CUDA is not available. Training will run on CPU. It is highly recommended to use a small --sample_fraction for quick debugging!")
    
    # 2. Set up the DataLoaders
    print(f"[INFO] Loading data splits...")
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    train_dataset = CatNonCatDataset(
        csv_file=os.path.join("metadata", "train_split.csv"),
        transform=train_transform,
        sample_fraction=args.sample_fraction
    )
    val_dataset = CatNonCatDataset(
        csv_file=os.path.join("metadata", "val_split.csv"),
        transform=val_transform,
        sample_fraction=args.sample_fraction
    )
    
    # Count samples per class and use a WeightedRandomSampler to counter class imbalance
    targets = train_dataset.df["label"].values
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in range(5)])
    weight = 1. / (class_sample_count + 1e-6)
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Train samples size: {len(train_dataset)} | Validation samples size: {len(val_dataset)}")
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # 3. Build the model
    print(f"[INFO] Initializing model: '{args.model}'...")
    if args.model == "baseline":
        model = get_baseline_model()
    elif args.model == "transfer":
        # With --fine_tune, unfreeze the whole backbone
        freeze_bb = not args.fine_tune
        # Backbone is chosen with --arch (resnet50, convnext_tiny or mobilenet_v2)
        model = get_transfer_model(args.arch, freeze_backbone=freeze_bb)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model = model.to(device)
    
    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Loading custom weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    # 4. Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    
    # Discriminative learning rates
    if args.fine_tune and args.model == "transfer":
        # Backbone gets a 10x lower learning rate than the classification head
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]
        optimizer_grouped_parameters = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr}
        ]
        if args.optimizer == "adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(optimizer_grouped_parameters, momentum=0.9, weight_decay=1e-4)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "adam":
            optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
            
    # Cosine annealing learning-rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
    # 5. Training loop
    print(f"[INFO] Start training for {args.epochs} epochs...")
    
    # CutMix and MixUp augmentation (applied per batch)
    cutmix = v2.CutMix(alpha=1.0, num_classes=5)
    mixup = v2.MixUp(alpha=0.2, num_classes=5)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Apply MixUp/CutMix (turns hard labels into soft labels)
            if not args.no_aug:
                images, labels = cutmix_or_mixup(images, labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            
            if not args.no_aug:
                # Soft labels: compare against the argmax class
                correct += predicted.eq(labels.argmax(dim=1)).sum().item()
            else:
                # Hard labels
                correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                acc = 100. * correct / total
                avg_loss = running_loss / total
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
                
        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        print(f"==> Epoch [{epoch+1}/{args.epochs}] Complete | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # Cosine annealing steps once per epoch (no loss needed)
        scheduler.step()
        
        # Save the best weights (lowest validation loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"best_{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[SAVED] Saved new best model checkpoint to '{model_path}'")
            
    # 6. Plot and save the curves
    plot_curves(history, args.model)
    print("[INFO] Training finished!")

def plot_curves(history, model_name):
    """
    Plot loss and accuracy curves and save them as an image.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"{model_name}_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved training curves plot to '{plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training Script")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "transfer"], help="Model to train")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "convnext_tiny", "mobilenet_v2"], help="Backbone architecture for --model transfer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (keep low for laptops)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of dataset to sample (0.0 to 1.0) for fast CPU run")
    parser.add_argument("--fine_tune", action="store_true", help="If flag is set, unfreeze the backbone for full fine-tuning")
    parser.add_argument("--no_aug", action="store_true", help="Disable MixUp/CutMix data augmentation")
    parser.add_argument("--weights", type=str, default="", help="Path to custom weights (.pth) to load before training")
    
    args = parser.parse_args()
    
    train_model(args)

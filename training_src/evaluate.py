import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the project dataset and models
from dataset import CatNonCatDataset, get_transforms
from baseline_model import get_baseline_model
from transfer_model import get_transfer_model

def evaluate_model(args):
    # 1. Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device for evaluation: {device}")
    
    # 2. Load the evaluation split
    test_transform = get_transforms(is_train=False)
    test_dataset = CatNonCatDataset(
        csv_file=os.path.join("metadata", f"{args.split}_split.csv"),
        transform=test_transform,
        sample_fraction=args.sample_fraction
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[INFO] Test dataset size: {len(test_dataset)} samples")
    
    # 3. Build the model and load its weights
    print(f"[INFO] Loading model structure: '{args.model}'...")
    if args.model == "baseline":
        model = get_baseline_model()
        weight_filename = "best_baseline.pth"
    elif args.model == "transfer":
        # The transfer model is ResNet50; the architecture must match the saved weights
        model = get_transfer_model("resnet50", freeze_backbone=True)
        weight_filename = "best_transfer.pth"
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model_path = os.path.join("models", weight_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Trained weights file not found at '{model_path}'! Please run training first.")
        
    print(f"[INFO] Loading weights from '{model_path}'...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Run predictions
    all_preds = []
    all_labels = []
    
    print("[INFO] Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 5. Compute metrics
    print("\n[INFO] =============== EVALUATION REPORT =============== ")
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"], 
        digits=4
    )
    print(report)
    
    # 6. Compute and plot the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(cm, args.model)

def plot_confusion_matrix(cm, model_name):
    """
    Plot the confusion matrix as a heatmap and save it.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"], 
        yticklabels=["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"]
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{model_name} Confusion Matrix")
    
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[INFO] Saved confusion matrix heatmap to '{plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Evaluation Script")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "transfer"], help="Model to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split CSV in metadata/ to evaluate on (preprocess.py writes train/val only)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of test dataset to sample (0.0 to 1.0) for fast CPU run")
    
    args = parser.parse_args()
    evaluate_model(args)

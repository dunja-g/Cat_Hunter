import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 导入自定义的数据集与模型
from dataset import CatNonCatDataset, get_transforms
from baseline_model import get_baseline_model
from transfer_model import get_transfer_model

def train_model(args):
    # 1. 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    if device.type == "cpu":
        print("[WARNING] CUDA is not available. Training will run on CPU. It is highly recommended to use a small --sample_fraction for quick debugging!")
    
    # 2. 设置数据加载器 (DataLoader)
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Train samples size: {len(train_dataset)} | Validation samples size: {len(val_dataset)}")
    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # 3. 初始化模型
    print(f"[INFO] Initializing model: '{args.model}'...")
    if args.model == "baseline":
        model = get_baseline_model()
    elif args.model == "transfer":
        model = get_transfer_model("mobilenet_v2", freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model = model.to(device)
    
    # 4. 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    
    # 过滤出需要更新梯度的参数 (用于迁移学习)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(trainable_params, lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    # 5. 开始训练循环
    print(f"[INFO] Start training for {args.epochs} epochs...")
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
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                acc = 100. * correct / total
                avg_loss = running_loss / total
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
                
        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total
        
        # 验证集评估
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
        
        # 记录历史数据
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # 保存最佳模型权重
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"best_{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[SAVED] Saved new best model checkpoint to '{model_path}'")
            
    # 6. 绘制并保存曲线
    plot_curves(history, args.model)
    print("[INFO] Training finished!")

def plot_curves(history, model_name):
    """
    绘制 Loss 和 Accuracy 对比曲线并保存为本地图像
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # 绘制 Accuracy 曲线
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
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (keep low for laptops)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of dataset to sample (0.0 to 1.0) for fast CPU run")
    
    args = parser.parse_args()
    
    train_model(args)

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the 5 target breeds
BREEDS = ["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"]

def create_splits(data_dir=os.path.join("data", "raw"), output_dir="metadata", seed=42):
    print("[INFO] Scanning image files for 5 cat breeds...")
    
    all_images = []
    all_labels = []
    
    for label_idx, breed in enumerate(BREEDS):
        breed_dir = os.path.join(data_dir, breed)
        # 支持多种常见图片格式
        patterns = [
            os.path.join(breed_dir, "*.jpg"), 
            os.path.join(breed_dir, "*.jpeg"), 
            os.path.join(breed_dir, "*.png"),
            os.path.join(breed_dir, "*.JPG")
        ]
        
        images = []
        for pattern in patterns:
            images.extend(glob.glob(pattern))
            
        print(f"[INFO] Found {len(images)} images for {breed}")
        all_images.extend(images)
        all_labels.extend([label_idx] * len(images))
        
    if len(all_images) == 0:
        print("[WARNING] No images found! Please ensure data is in data/raw/<BreedName>/")
        return
        
    df = pd.DataFrame({"image_path": all_images, "label": all_labels})
    
    # 按照任务要求，划分为 85% 训练集，15% 验证集 (没有专门的测试集，测试集是实车摄像头)
    # Stratified 确保 5 种猫的比例在训练集和验证集中都均衡
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["label"])
    
    print("\n[INFO] Dataset split statistics (85/15):")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val:   {len(val_df)} samples")
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    
    print(f"\n[INFO] Dataset splits CSV saved to '{output_dir}/' successfully!")

if __name__ == "__main__":
    create_splits()

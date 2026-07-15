import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(archive_dir, non_cats_dir, output_dir="metadata", seed=42):
    """
    扫描猫和非猫的图像，将其划分为训练集 (70%)、验证集 (15%) 和测试集 (15%)，
    并将划分结果保存到 CSV 文件中。
    """
    print("[INFO] Scanning image files...")
    
    # 1. 扫描猫图像 (Class 1)
    # 注意：我们优先扫描根目录下的 CAT_00 到 CAT_06，避免扫描重复的 cats 文件夹
    cat_patterns = [
        os.path.join(archive_dir, "CAT_0[0-6]", "*.jpg"),
    ]
    cat_images = []
    for pattern in cat_patterns:
        cat_images.extend(glob.glob(pattern))
    
    # 如果根目录下找不到，再尝试 cats 子目录
    if len(cat_images) == 0:
        cat_patterns_fallback = [
            os.path.join(archive_dir, "cats", "CAT_0[0-6]", "*.jpg"),
        ]
        for pattern in cat_patterns_fallback:
            cat_images.extend(glob.glob(pattern))
            
    print(f"[INFO] Cat images found: {len(cat_images)}")
    
    # 2. 扫描非猫图像 (Class 0)
    non_cat_patterns = [
        os.path.join(non_cats_dir, "*", "*.jpg"),
    ]
    non_cat_images = []
    for pattern in non_cat_patterns:
        non_cat_images.extend(glob.glob(pattern))
        
    print(f"[INFO] Non-cat (ship) images found: {len(non_cat_images)}")
    
    if len(cat_images) == 0 or len(non_cat_images) == 0:
        raise ValueError("[ERROR] Cat or non-cat image count is 0. Please check dataset path!")
        
    # 3. 构建 DataFrame 并分配标签
    cat_df = pd.DataFrame({"image_path": cat_images, "label": 1})
    non_cat_df = pd.DataFrame({"image_path": non_cat_images, "label": 0})
    
    # 4. 数据集划分 (70/15/15)
    # 首先，把猫和非猫的数据分别划分，保证正负样本的比例在各个集合中一致 (Stratified Split)
    # 第一步：划分出 70% 训练集，和 30% 临时集 (验证+测试)
    cat_train, cat_temp = train_test_split(cat_df, test_size=0.30, random_state=seed)
    non_cat_train, non_cat_temp = train_test_split(non_cat_df, test_size=0.30, random_state=seed)
    
    # 第二步：将 30% 临时集平分为 15% 验证集和 15% 测试集
    cat_val, cat_test = train_test_split(cat_temp, test_size=0.50, random_state=seed)
    non_cat_val, non_cat_test = train_test_split(non_cat_temp, test_size=0.50, random_state=seed)
    
    # 合并猫与非猫的数据
    train_df = pd.concat([cat_train, non_cat_train]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([cat_val, non_cat_val]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([cat_test, non_cat_test]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    print("\n[INFO] Dataset split statistics:")
    print(f"  - Train: {len(train_df)} (Cat: {sum(train_df['label'] == 1)}, Non-cat: {sum(train_df['label'] == 0)})")
    print(f"  - Val:   {len(val_df)} (Cat: {sum(val_df['label'] == 1)}, Non-cat: {sum(val_df['label'] == 0)})")
    print(f"  - Test:  {len(test_df)} (Cat: {sum(test_df['label'] == 1)}, Non-cat: {sum(test_df['label'] == 0)})")
    
    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"\n[INFO] Dataset splits CSV saved to '{output_dir}/' successfully!")

if __name__ == "__main__":
    archive_path = os.path.abspath("archive")
    non_cats_path = os.path.abspath("non_cats")
    
    create_splits(archive_path, non_cats_path)

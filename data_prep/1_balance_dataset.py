import os
import random
import shutil
import subprocess

TARGET_COUNT = 400
REPO_URL = "https://github.com/12412825-collab/5-cat-category-dataset.git"
TEMP_DIR = "temp_unbalanced_data"
RAW_DIR = os.path.join("data", "raw")

def main():
    print("==================================================")
    print("  Step 1: Download & Balance Existing Data")
    print("==================================================")

    # 1. Clone the repo if we haven't already
    if not os.path.exists(TEMP_DIR):
        print(f"[INFO] Cloning your teammate's repo into {TEMP_DIR}...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, TEMP_DIR])
    else:
        print(f"[INFO] {TEMP_DIR} already exists, skipping clone.")

    # 2. Create target directories
    os.makedirs(RAW_DIR, exist_ok=True)
    breeds = ["Persian", "Ragdoll", "Sphynx", "Pallas", "Singapura"]
    for b in breeds:
        os.makedirs(os.path.join(RAW_DIR, b), exist_ok=True)

    # 3. Process each breed
    for breed in breeds:
        source_dir = os.path.join(TEMP_DIR, breed)
        target_dir = os.path.join(RAW_DIR, breed)
        
        if not os.path.exists(source_dir):
            print(f"[WARN] Source directory {source_dir} not found!")
            continue

        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # If it has more than TARGET_COUNT, downsample it randomly
        if len(images) > TARGET_COUNT:
            print(f"[INFO] {breed}: Downsampling from {len(images)} to {TARGET_COUNT}...")
            # We seed the random number generator so the "random" selection is the same every time you run it
            random.seed(42)
            selected = random.sample(images, TARGET_COUNT)
        else:
            print(f"[INFO] {breed}: Copying all {len(images)} existing images (needs {TARGET_COUNT - len(images)} more later)...")
            selected = images

        # Copy the selected images over
        copied_count = 0
        for img in selected:
            src = os.path.join(source_dir, img)
            dst = os.path.join(target_dir, img)
            # Only copy if it doesn't already exist to save time
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied_count += 1

        print(f"       -> Copied {copied_count} images to {target_dir}")

    print("\n[SUCCESS] Step 1 Complete! Data is safely downsampled and isolated in your data/raw/ folder.")

if __name__ == "__main__":
    main()

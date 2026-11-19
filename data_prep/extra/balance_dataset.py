import os
import random
import shutil
import subprocess
import sys
import glob

# Ensure bing-image-downloader is installed
try:
    import bing_image_downloader
except ImportError:
    print("[INFO] Installing bing-image-downloader...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])
    
from bing_image_downloader import downloader

TARGET_COUNT = 1000
DATA_DIR = os.path.join("data", "raw")

BREEDS = {
    "Persian": "Persian cat",
    "Ragdoll": "Ragdoll cat",
    "Sphynx": "Sphynx cat",
    "Pallas": "Pallas cat",
    "Singapura": "Singapura cat"
}

def get_images(folder):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    return list(set(images))

def downsample_class(breed, current_images):
    excess = len(current_images) - TARGET_COUNT
    print(f"[INFO] {breed} has {len(current_images)} images. Randomly deleting {excess} images to reach {TARGET_COUNT}.")
    
    # Shuffle and select images to delete
    random.seed(42)
    images_to_delete = random.sample(current_images, excess)
    
    for img_path in images_to_delete:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"[WARNING] Could not delete {img_path}: {e}")
            
    print(f"[SUCCESS] {breed} downsampled to {TARGET_COUNT} images.")

def upsample_class(breed, search_query, current_count):
    deficit = TARGET_COUNT - current_count
    # Bing downloader sometimes fails on some links, so we request a little more to be safe
    download_limit = int(deficit * 1.1) 
    
    print(f"[INFO] {breed} has {current_count} images. Need {deficit} more. Downloading up to {download_limit} images using query: '{search_query}'...")
    
    temp_dir = os.path.join("data", "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        downloader.download(
            query=search_query,
            limit=download_limit,
            output_dir=temp_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] Failed during download for {breed}: {e}")
        
    # Move downloaded images to the actual breed directory
    downloaded_folder = os.path.join(temp_dir, search_query)
    if os.path.exists(downloaded_folder):
        downloaded_images = get_images(downloaded_folder)
        print(f"[INFO] Downloaded {len(downloaded_images)} images for {breed}. Moving to {DATA_DIR}/{breed}...")
        
        target_folder = os.path.join(DATA_DIR, breed)
        moved_count = 0
        for img in downloaded_images:
            if current_count + moved_count >= TARGET_COUNT:
                break # Reached exactly TARGET_COUNT
                
            filename = os.path.basename(img)
            # Prepend 'bing_' to avoid name collisions
            new_path = os.path.join(target_folder, f"bing_{moved_count}_{filename}")
            try:
                shutil.move(img, new_path)
                moved_count += 1
            except Exception as e:
                pass
                
        print(f"[SUCCESS] Moved {moved_count} new images for {breed}. Total is now {current_count + moved_count}.")
        
    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def main():
    print("==================================================")
    print(f"Dataset Balancer: Targeting {TARGET_COUNT} images per class")
    print("==================================================")
    
    for breed, search_query in BREEDS.items():
        folder = os.path.join(DATA_DIR, breed)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        current_images = get_images(folder)
        count = len(current_images)
        
        if count > TARGET_COUNT:
            downsample_class(breed, current_images)
        elif count < TARGET_COUNT:
            upsample_class(breed, search_query, count)
        else:
            print(f"[INFO] {breed} already has exactly {TARGET_COUNT} images. Skipping.")
            
    print("\n[DONE] Dataset balancing operations completed.")

if __name__ == "__main__":
    main()

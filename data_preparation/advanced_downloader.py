import os
import glob
import urllib.request
import socket
import time
import subprocess
import sys
import shutil

# Ensure required libraries
try:
    from duckduckgo_search import DDGS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "duckduckgo_search"])
    from duckduckgo_search import DDGS

try:
    from bing_image_downloader import downloader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])
    from bing_image_downloader import downloader

DATA_DIR = os.path.join("data", "raw")
TARGET = 1000

# Setting up multiple query variations to bypass single-query pagination limits
QUERIES = {
    "Sphynx": [
        "Sphynx cat", "Hairless cat", "Sphynx kitten", 
        "Sphynx cat playing", "Sphynx cat portrait", "Canadian Sphynx"
    ],
    "Pallas": [
        "Pallas cat", "Manul", "Otocolobus manul", 
        "Pallas's cat", "Manul kitten", "Wild Pallas cat"
    ],
    "Singapura": [
        "Singapura cat", "Singapura kitten", "Singapura breed", 
        "Smallest cat breed Singapura", "Singapura cat portrait"
    ]
}

def get_current_count(breed):
    folder = os.path.join(DATA_DIR, breed)
    if not os.path.exists(folder):
        return 0
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    return len(set(images))

def download_image(url, save_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as response, open(save_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        return False

def duckduckgo_download(breed, query, count_needed):
    if count_needed <= 0: return 0
    print(f"    [DDG] Searching DuckDuckGo for: '{query}'...")
    downloaded = 0
    folder = os.path.join(DATA_DIR, breed)
    os.makedirs(folder, exist_ok=True)
    
    try:
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=count_needed + 50)
            if not results: return 0
            
            for res in results:
                url = res.get("image")
                if not url: continue
                
                ext = url.split(".")[-1].split("?")[0].lower()
                if ext not in ["jpg", "jpeg", "png"]:
                    ext = "jpg"
                    
                filename = f"ddg_{int(time.time() * 1000)}_{downloaded}.{ext}"
                save_path = os.path.join(folder, filename)
                
                if download_image(url, save_path):
                    downloaded += 1
                    if downloaded % 20 == 0:
                        print(f"      Downloaded {downloaded} images from DDG...")
                
                if downloaded >= count_needed:
                    break
    except Exception as e:
        print(f"    [DDG] Error: {e}")
        
    return downloaded

def main():
    socket.setdefaulttimeout(5)
    print("==================================================")
    print("Advanced Downloader: Attempting to reach 1000 images")
    print("==================================================")
    
    for breed, queries in QUERIES.items():
        count = get_current_count(breed)
        print(f"\n[INFO] {breed} has {count} images.")
        
        for query in queries:
            count = get_current_count(breed)
            needed = TARGET - count
            if needed <= 0:
                print(f"[SUCCESS] {breed} reached {TARGET} images!")
                break
                
            print(f"  -> Need {needed} more images. Trying query: '{query}'")
            
            # 1. Try DuckDuckGo first
            added = duckduckgo_download(breed, query, needed)
            count = get_current_count(breed)
            needed = TARGET - count
            
            if needed <= 0:
                break
                
            # 2. Try Bing Image Downloader
            temp_dir = os.path.join("data", "temp_downloads")
            try:
                downloader.download(
                    query=query, limit=min(needed, 100), output_dir=temp_dir,
                    adult_filter_off=True, force_replace=False, timeout=5, verbose=False
                )
                
                downloaded_folder = os.path.join(temp_dir, query)
                if os.path.exists(downloaded_folder):
                    images = []
                    for ext in ("*.jpg", "*.jpeg", "*.png"):
                        images.extend(glob.glob(os.path.join(downloaded_folder, ext)))
                        images.extend(glob.glob(os.path.join(downloaded_folder, ext.upper())))
                    
                    target_folder = os.path.join(DATA_DIR, breed)
                    moved = 0
                    for img in images:
                        if needed - moved <= 0: break
                        filename = os.path.basename(img)
                        new_path = os.path.join(target_folder, f"bing_{int(time.time()*1000)}_{moved}_{filename}")
                        try:
                            shutil.move(img, new_path)
                            moved += 1
                        except: pass
                    print(f"    [BING] Saved {moved} new images from Bing.")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"    [BING] Error: {e}")
                
        final_count = get_current_count(breed)
        print(f"[SUMMARY] {breed} now has {final_count} images.")

if __name__ == "__main__":
    main()

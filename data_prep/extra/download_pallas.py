import os
import subprocess
import sys
import time

def install_and_download():
    print("==============================================")
    print("Pallas Cat Image Downloader (using Bing)")
    print("==============================================")
    
    # 1. Install the required library automatically
    print("\n[INFO] Checking for and installing the bing-image-downloader library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])
    
    from bing_image_downloader import downloader
    
    # 2. Set the output path
    # Save under data/raw to match the project layout
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Start the bulk download
    print("\n[INFO] Downloading 300 Pallas cat images from Bing...")
    print("[INFO] This can take a few minutes. Some links will fail during download; that is normal and they are skipped automatically.")
    time.sleep(2)
    
    try:
        downloader.download(
            query="Pallas cat", 
            limit=300,  
            output_dir=output_dir, 
            adult_filter_off=True, 
            force_replace=False, 
            timeout=10,
            verbose=False
        )
    except Exception as e:
        print(f"\n[ERROR] Error during download: {e}")
        
    # 4. Rename the folder to Pallas
    # The downloader creates a folder named after the search query ("Pallas cat") by default
    old_folder = os.path.join(output_dir, 'Pallas cat')
    new_folder = os.path.join(output_dir, 'Pallas')
    
    if os.path.exists(old_folder):
        if not os.path.exists(new_folder):
            os.rename(old_folder, new_folder)
            print(f"\n[SUCCESS] All images downloaded and renamed; saved in: {new_folder}")
        else:
            print(f"\n[SUCCESS] Download complete. Images saved in: {old_folder} (because {new_folder} already exists)")

if __name__ == "__main__":
    install_and_download()

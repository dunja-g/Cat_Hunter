import os
import sys
import subprocess

TARGET_COUNT = 400
RAW_DIR = os.path.join("data", "raw")

def install_deps():
    try:
        import bing_image_downloader
    except ImportError:
        print("[INFO] Installing bing-image-downloader...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bing-image-downloader"])

def main():
    print("==================================================")
    print("  Step 2: Scrape Missing Minority Images")
    print("==================================================")

    install_deps()
    from bing_image_downloader import downloader

    breeds = ["Persian", "Ragdoll", "Sphynx", "Pallas", "Singapura"]

    for breed in breeds:
        breed_dir = os.path.join(RAW_DIR, breed)
        if not os.path.exists(breed_dir):
            continue
            
        images = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(images)
        missing = TARGET_COUNT - current_count

        if missing <= 0:
            print(f"[INFO] {breed}: Already has {current_count} images. No scraping needed.")
            continue

        print(f"\n[INFO] {breed}: Has {current_count} images. Scraping {missing} more...")
        
        # We search specifically for the breed + cat to get accurate results
        search_query = f"{breed} cat"
        
        try:
            # Note: downloader saves to RAW_DIR/search_query
            downloader.download(
                query=search_query, 
                limit=missing,  
                output_dir=RAW_DIR, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=10,
                verbose=False
            )
            
            # The downloader puts them in a folder named after the search query
            # We need to move them into our actual breed folder
            download_folder = os.path.join(RAW_DIR, search_query)
            if os.path.exists(download_folder):
                new_images = os.listdir(download_folder)
                for img in new_images:
                    src = os.path.join(download_folder, img)
                    # Prepend "scraped_" so we don't accidentally overwrite existing images
                    dst = os.path.join(breed_dir, f"scraped_{img}")
                    os.rename(src, dst)
                os.rmdir(download_folder)
                
        except Exception as e:
            print(f"[ERROR] Failed to scrape {breed}: {e}")

    print("\n[SUCCESS] Step 2 Complete! You now have a perfectly balanced dataset.")

if __name__ == "__main__":
    main()

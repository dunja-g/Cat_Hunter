import csv
import glob
import os
import random


BREEDS = ["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"]
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _collect_images(breed_dir):
    images = []
    for pattern in IMAGE_PATTERNS:
        images.extend(glob.glob(os.path.join(breed_dir, pattern)))
    return sorted(set(images))


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def create_splits(data_dir=os.path.join("data", "raw"), output_dir="metadata", seed=42):
    print("[INFO] Scanning image files for 5 cat breeds...")

    rng = random.Random(seed)
    train_rows = []
    val_rows = []

    for label_idx, breed in enumerate(BREEDS):
        breed_dir = os.path.join(data_dir, breed)
        images = _collect_images(breed_dir)
        rng.shuffle(images)

        print(f"[INFO] Found {len(images)} images for {breed}")
        if not images:
            continue

        val_count = max(1, round(len(images) * 0.15)) if len(images) > 1 else 0
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_rows.extend({"image_path": path, "label": label_idx} for path in train_images)
        val_rows.extend({"image_path": path, "label": label_idx} for path in val_images)

    if not train_rows and not val_rows:
        print("[WARNING] No images found! Please ensure data is in data/raw/<BreedName>/")
        return

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    os.makedirs(output_dir, exist_ok=True)
    _write_csv(os.path.join(output_dir, "train_split.csv"), train_rows)
    _write_csv(os.path.join(output_dir, "val_split.csv"), val_rows)

    print("\n[INFO] Dataset split statistics (85/15):")
    print(f"  - Train: {len(train_rows)} samples")
    print(f"  - Val:   {len(val_rows)} samples")
    print(f"\n[INFO] Dataset splits CSV saved to '{output_dir}/' successfully!")


if __name__ == "__main__":
    create_splits()

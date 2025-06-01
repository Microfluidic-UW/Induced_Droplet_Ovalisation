import os
import shutil
import random
from pathlib import Path

# Configuration
IMAGE_ROOT = Path("train_data/images")
LABEL_ROOT = Path("train_data/labels")
OUTPUT_ROOT = Path("ready_for_training")
SPLITS = {
    "training": 0.8,
    "validation": 0.2,
}
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# Step 1: Collect all image paths
all_image_paths = [p for p in IMAGE_ROOT.rglob("*") if p.suffix.lower() in VALID_IMAGE_EXTS]

# Step 2: Shuffle and split
random.shuffle(all_image_paths)
total = len(all_image_paths)
split_index = int(total * SPLITS["training"])
split_paths = {
    "training": all_image_paths[:split_index],
    "validation": all_image_paths[split_index:],
}

# Step 3: Copy images and matching labels
for split, paths in split_paths.items():
    for image_path in paths:
        rel_path = image_path.relative_to(IMAGE_ROOT)
        label_path = LABEL_ROOT / rel_path.with_suffix(".txt")
        
        # Destination paths
        image_dest = OUTPUT_ROOT / split / "images" / rel_path
        label_dest = OUTPUT_ROOT / split / "labels" / rel_path.with_suffix(".txt")
        
        # Create necessary directories
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        shutil.copy2(image_path, image_dest)
        
        # Copy label if it exists
        if label_path.exists():
            shutil.copy2(label_path, label_dest)
        else:
            print(f"⚠️ Label not found for image: {rel_path}")

print("✅ Dataset split and copied successfully!")

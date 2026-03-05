"""Scan dataset for corrupted or unreadable images and optionally remove them."""
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image

# Suppress PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None


def check_image(img_path: str) -> tuple[str, str] | None:
    """Check if an image file is valid. Returns (path, error) if corrupt, else None."""
    try:
        img = Image.open(img_path)
        img.verify()  # verify file integrity
    except Exception as e:
        return (img_path, f"verify failed: {e}")

    try:
        img = Image.open(img_path)
        img.load()  # actually decode the pixel data
    except Exception as e:
        return (img_path, f"load failed: {e}")

    # Check file size (0-byte files)
    if os.path.getsize(img_path) == 0:
        return (img_path, "empty file (0 bytes)")

    return None


def find_all_images(root_dir: str) -> list[str]:
    """Recursively find all image files."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(dirpath, f))
    return images


def main():
    dataset_root = "/mnt/d/AeroDefence/augmented"

    print(f"Scanning dataset at: {dataset_root}")
    print("Finding all image files...")

    images = find_all_images(dataset_root)
    print(f"Found {len(images)} images. Validating...")

    # Use multiprocessing for speed
    workers = min(cpu_count(), 8)
    corrupt = []
    checked = 0

    with Pool(workers) as pool:
        for result in pool.imap_unordered(check_image, images, chunksize=50):
            checked += 1
            if checked % 500 == 0:
                print(f"  Checked {checked}/{len(images)}...", flush=True)
            if result is not None:
                corrupt.append(result)
                print(f"  CORRUPT: {result[0]} — {result[1]}", flush=True)

    print(f"\n{'='*60}")
    print(f"Results: {len(corrupt)} corrupt out of {len(images)} total images")
    print(f"{'='*60}")

    if not corrupt:
        print("All images are valid!")
        return

    # Print list of corrupt files
    print("\nCorrupt files:")
    for path, error in corrupt:
        print(f"  {path}")
        print(f"    Error: {error}")

    # Ask to remove
    print(f"\nTo remove corrupt images and their labels, run:")
    print(f"  python {sys.argv[0]} --remove")

    if "--remove" in sys.argv:
        print("\nRemoving corrupt images and associated labels...")
        for path, _ in corrupt:
            # Remove image
            try:
                os.remove(path)
                print(f"  Removed image: {path}")
            except OSError as e:
                print(f"  Failed to remove {path}: {e}")

            # Remove corresponding label file
            label_path = path.replace("/images/", "/labels/")
            for ext in ['.txt']:
                lp = str(Path(label_path).with_suffix(ext))
                if os.path.exists(lp):
                    try:
                        os.remove(lp)
                        print(f"  Removed label: {lp}")
                    except OSError as e:
                        print(f"  Failed to remove {lp}: {e}")

        print(f"\nDone! Removed {len(corrupt)} corrupt images and their labels.")


if __name__ == "__main__":
    main()

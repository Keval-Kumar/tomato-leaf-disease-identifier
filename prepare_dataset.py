import random
import shutil
import time
from pathlib import Path

from PIL import Image
from PIL import UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_DIR = BASE_DIR / "data" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

BAD_DIR = BASE_DIR / "data" / "_bad_images"


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def prepare_dataset(val_ratio: float = 0.2, seed: int = 42):
    if not RAW_DIR.exists():
        raise SystemExit(
            f"RAW_DIR {RAW_DIR} does not exist.\n"
            "Place your original downloaded tomato leaf images under data/raw/\n"
            "in subfolders per class (e.g. data/raw/bacterial_spot, data/raw/healthy, ...)."
        )

    random.seed(seed)

    # Clean old dataset folders
    def safe_rmtree(path: Path):
        if not path.exists():
            return
        for _ in range(5):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                time.sleep(0.2)
        # Last try: ignore errors (locked file on Windows)
        shutil.rmtree(path, ignore_errors=True)

    safe_rmtree(TRAIN_DIR)
    safe_rmtree(VAL_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No class folders found inside {RAW_DIR}.")

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not images:
            print(f"Skipping {class_name}: no images found.")
            continue
        good_images = []
        for img in images:
            if is_valid_image(img):
                good_images.append(img)
            else:
                BAD_DIR.mkdir(parents=True, exist_ok=True)
                bad_dest = BAD_DIR / class_name
                bad_dest.mkdir(parents=True, exist_ok=True)
                # On Windows, the file can be briefly locked by another process.
                # If moving fails, we just skip it and continue.
                try:
                    shutil.move(str(img), str(bad_dest / img.name))
                except PermissionError:
                    pass
        images = good_images
        if not images:
            print(f"Skipping {class_name}: no valid images found.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_ratio))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_class_dir / img.name)

        print(
            f"Class '{class_name}': {len(train_imgs)} train images, {len(val_imgs)} val images "
            f"(total {len(images)})"
        )

    print("Dataset prepared.")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Val directory:   {VAL_DIR}")


if __name__ == "__main__":
    prepare_dataset()


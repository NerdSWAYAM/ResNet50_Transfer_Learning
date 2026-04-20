import os
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
SPLITS       = ["train/Wheat leaf blight", "val/Wheat leaf blight"]          # folders to process
ROOT_DIR     = "."                         # set to your dataset root path
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Which flips to generate (set False to skip any)
DO_HFLIP     = True   # horizontal flip  → _hflip
DO_VFLIP     = False   # vertical flip    → _vflip
DO_BOTH_FLIP = False   # both axes flip   → _hvflip
# ────────────────────────────────────────────────────────────────────────────


def augment_folder(folder_path: str) -> int:
    """Augments all images in folder_path (including class subfolders).
    Returns the number of new images saved."""
    saved = 0

    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTENSIONS:
                continue

            img_path = os.path.join(dirpath, fname)
            stem     = os.path.splitext(fname)[0]
            img      = Image.open(img_path)

            flips = []
            if DO_HFLIP:
                flips.append((img.transpose(Image.FLIP_LEFT_RIGHT), f"{stem}_hflip{ext}"))
            if DO_VFLIP:
                flips.append((img.transpose(Image.FLIP_TOP_BOTTOM),  f"{stem}_vflip{ext}"))
            if DO_BOTH_FLIP:
                both = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                flips.append((both, f"{stem}_hvflip{ext}"))

            for aug_img, aug_name in flips:
                save_path = os.path.join(dirpath, aug_name)
                if not os.path.exists(save_path):   # skip if already exists
                    aug_img.save(save_path)
                    saved += 1

    return saved


def main():
    for split in SPLITS:
        split_path = os.path.join(ROOT_DIR, split)

        if not os.path.isdir(split_path):
            print(f"[SKIP] '{split_path}' not found")
            continue

        print(f"[{split.upper()}] Augmenting '{split_path}' ...")
        count = augment_folder(split_path)
        print(f"[{split.upper()}] Done — {count} new images saved\n")


if __name__ == "__main__":
    main()
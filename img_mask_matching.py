## Verify that every image has a corresponding mask

from utils import PROCESSED_BRACKET_BLACK_DATA_DIR

splits = ["train", "val"]

for split in splits:
    img_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / split / "images"
    mask_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / split / "masks"
    missing = []
    for img_path in img_dir.glob("*.png"):
        base_name = img_path.stem  # e.g., 000_good, 000_hole, 000_scratch
        if base_name.endswith(("good", "hole", "scratch")):
            mask_name = f"{base_name}_mask.png"
            mask_path = mask_dir / mask_name
            if not mask_path.exists():
                missing.append(mask_name)
    print(f"\n{split.capitalize()} set: {len(missing)} missing masks")
    if missing:
        print("Missing masks:", missing)
    else:
        print("All masks found for images in", img_dir)

    ## Expected output

#     Train set: 0 missing masks
# All masks found for images in C:\Users\Windows\Downloads\defect_detection\data\processed\bracket_black\dataset\train\images
#
# Val set: 0 missing masks
# All masks found for images in C:\Users\Windows\Downloads\defect_detection\data\processed\bracket_black\dataset\val\images
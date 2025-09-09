import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from utils import DATA_DIR, PROCESSED_DATA_DIR, BRACKET_BLACK_DATA_DIR, PROCESSED_BRACKET_BLACK_DATA_DIR

ROOT = DATA_DIR       # full dataset root
OUT = PROCESSED_DATA_DIR       # output dataset root
OUT.mkdir(exist_ok=True)

# Generic function to copy and rename images with a suffix

def copy_and_rename_images(src_dir, dst_dir, suffix):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.glob("*.png"):
        base_name = os.path.splitext(img_path.name)[0]
        new_name = f"{base_name}_{suffix}.png"
        shutil.copy(img_path, dst_dir/new_name)
        print(f"Copied {img_path} -> {dst_dir/new_name}")


# Generic function to copy and rename mask files with a custom suffix
def copy_and_rename_masks(src_dir, dst_dir, suffix):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in src_dir.glob("*.png"):
        base_name = os.path.splitext(mask_path.name)[0]
        # If base_name ends with _mask, replace with _<suffix>_mask
        if base_name.endswith("_mask"):
            new_base = base_name[:-5] + f"_{suffix}_mask"
        else:
            new_base = base_name + f"_{suffix}_mask"
        print(f"Base name: {base_name} -> New base: {new_base}")
        new_name = f"{new_base}.png"
        shutil.copy(mask_path, dst_dir/new_name)
        print(f"Copied {mask_path} -> {dst_dir/new_name}")

if __name__ == "__main__":

    # '''Rename images in test/hole and test/scratches''' 
    # hole_dir = BRACKET_BLACK_DATA_DIR/"test/hole"
    # scratch_dir = BRACKET_BLACK_DATA_DIR/"test/scratches"
    # # Copy and rename hole images directly to processed dir
    # processed_hole_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"test/hole"
    # processed_hole_dir.mkdir(parents=True, exist_ok=True)
    # for img_path in hole_dir.glob("*.png"):
    #     base_name = os.path.splitext(img_path.name)[0]
    #     new_name = f"{base_name}_hole.png"
    #     shutil.copy(img_path, processed_hole_dir/new_name)
    #     print(f"Copied {img_path} -> {processed_hole_dir/new_name}")

    # # Copy and rename scratch images directly to processed dir
    # processed_scratch_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"test/scratches"
    # processed_scratch_dir.mkdir(parents=True, exist_ok=True)
    # for img_path in scratch_dir.glob("*.png"):
    #     base_name = os.path.splitext(img_path.name)[0]
    #     new_name = f"{base_name}_scratch.png"
    #     shutil.copy(img_path, processed_scratch_dir/new_name)
    #     print(f"Copied {img_path} -> {processed_scratch_dir/new_name}")

    '''Rename images in test/hole and test/scratches''' 
    hole_dir = BRACKET_BLACK_DATA_DIR/"test/hole"
    scratch_dir = BRACKET_BLACK_DATA_DIR/"test/scratches"
    processed_hole_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"test/hole"
    processed_scratch_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"test/scratches"

    copy_and_rename_images(hole_dir, processed_hole_dir, "hole")
    copy_and_rename_images(scratch_dir, processed_scratch_dir, "scratch")


    hole_mask_dir = BRACKET_BLACK_DATA_DIR/"ground_truth/hole"
    processed_hole_mask_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"ground_truth/hole"
    copy_and_rename_masks(hole_mask_dir, processed_hole_mask_dir, "hole")

    scratch_mask_dir = BRACKET_BLACK_DATA_DIR/"ground_truth/scratches"
    processed_scratch_mask_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"ground_truth/scratches"
    copy_and_rename_masks(scratch_mask_dir, processed_scratch_mask_dir, "scratch")
    

    test_good_dir = BRACKET_BLACK_DATA_DIR/"test/good"
    train_good_dir = BRACKET_BLACK_DATA_DIR/"train/good"

    processed_good_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"test/good"
    processed_good_dir.mkdir(parents=True, exist_ok=True)
    for img_path in test_good_dir.glob("*.png"):
        base_name = os.path.splitext(img_path.name)[0]
        new_name = f"{base_name}_good.png"
        shutil.copy(img_path, processed_good_dir/new_name)
        print(f"Copied {img_path} -> {processed_good_dir/new_name}")
    
    processed_train_good_dir = PROCESSED_BRACKET_BLACK_DATA_DIR/"train/good"
    processed_train_good_dir.mkdir(parents=True, exist_ok=True)
    for img_path in train_good_dir.glob("*.png"):
        base_name = os.path.splitext(img_path.name)[0]
        new_name = f"{base_name}_good.png"
        shutil.copy(img_path, processed_train_good_dir/new_name)
        print(f"Copied {img_path} -> {processed_train_good_dir/new_name}")

    random.seed(42)

    train_good = list((PROCESSED_BRACKET_BLACK_DATA_DIR/"train/good").glob("*.png"))
    test_good = list((PROCESSED_BRACKET_BLACK_DATA_DIR/"test/good").glob("*.png"))
    test_hole = list((PROCESSED_BRACKET_BLACK_DATA_DIR/"test/hole").glob("*.png"))
    test_scratch = list((PROCESSED_BRACKET_BLACK_DATA_DIR/"test/scratches").glob("*.png"))

    gt_root_hole = PROCESSED_BRACKET_BLACK_DATA_DIR/"ground_truth"/"hole"
    gt_root_scratch = PROCESSED_BRACKET_BLACK_DATA_DIR/"ground_truth"/"scratches"
    gt_root = PROCESSED_BRACKET_BLACK_DATA_DIR/"ground_truth"

    # Labels
    all_defects = [(str(p), "defect") for p in test_hole + test_scratch]
    print(f"Found {len(all_defects)} defect samples.")

    all_good = [(str(p), "good") for p in train_good + test_good]
    print(f"Found {len(all_good)} good samples.") 

    ## We apply stratifed split to maintain class balance

    # Split defects: 70% train, 30% val
    defect_train, defect_val = train_test_split(all_defects, test_size=0.3, random_state=42)
    print(f"Defect train: {len(defect_train)}, val: {len(defect_val)}")

    # # Split good images similarly 
    good_train, good_val = train_test_split(all_good, test_size=0.3, random_state=42)
    print(f"Good train: {len(good_train)}, val: {len(good_val)}")


    # Build splits
    train_split = defect_train + good_train
    print(f"Total train samples: {len(train_split)}")
    val_split = defect_val + good_val
    print(f"Total val samples: {len(val_split)}")

    # Shuffle to mix good/defect
    random.shuffle(train_split)
    random.shuffle(val_split)

    # print("train splitttt", train_split)
    ## ('C:\\Users\\Windows\\Downloads\\defect_detection\\data\\processed\\bracket_black\\train\\good\\235_good.png', 'good')
    # --- Save images and masks for train split ---
    train_img_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / "train" / "images"
    train_mask_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / "train" / "masks"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label in train_split:
        img_name = os.path.basename(img_path)  ### 235_good.png
        base_name = os.path.splitext(img_name)[0]   ### 235_good
        # Copy image
        dest_img = train_img_dir / img_name  
        shutil.copy(img_path, dest_img)
        # Map to mask
        if "good" in img_name:
            # Create black mask
            img = Image.open(img_path)
            w, h = img.size
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            mask_name = f"{base_name}_mask.png"
            Image.fromarray(zero_mask).save(train_mask_dir / mask_name)
        elif "hole" in img_name:
            mask_name = f"{base_name}_mask.png"
            mask_path = (PROCESSED_BRACKET_BLACK_DATA_DIR / "ground_truth" / "hole" / mask_name)
            if mask_path.exists():
                shutil.copy(mask_path, train_mask_dir / mask_name)
            else:
                print(f"Warning: Mask not found for {img_name}")
        elif "scratch" in img_name:
            mask_name = f"{base_name}_mask.png"
            mask_path = (PROCESSED_BRACKET_BLACK_DATA_DIR / "ground_truth" / "scratches" / mask_name)
            if mask_path.exists():
                shutil.copy(mask_path, train_mask_dir / mask_name)
            else:
                print(f"Warning: Mask not found for {img_name}")

    num_train_images = len(list(train_img_dir.glob("*.png")))
    num_train_masks = len(list(train_mask_dir.glob("*.png")))
    print(f"Total number of train images: {num_train_images}")
    print(f"Total number of train masks: {num_train_masks}")


        # --- Save images and masks for val split ---
    val_img_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / "val" / "images"
    val_mask_dir = PROCESSED_BRACKET_BLACK_DATA_DIR / "dataset" / "val" / "masks"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label in val_split:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        # Copy image
        dest_img = val_img_dir / img_name
        shutil.copy(img_path, dest_img)
        # Map to mask
        if "good" in img_name:
            # Create black mask
            img = Image.open(img_path)
            w, h = img.size
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            mask_name = f"{base_name}_mask.png"
            Image.fromarray(zero_mask).save(val_mask_dir / mask_name)
        elif "hole" in img_name:
            mask_name = f"{base_name}_mask.png"
            mask_path = (PROCESSED_BRACKET_BLACK_DATA_DIR / "ground_truth" / "hole" / mask_name)
            if mask_path.exists():
                shutil.copy(mask_path, val_mask_dir / mask_name)
            else:
                print(f"Warning: Mask not found for {img_name}")
        elif "scratch" in img_name:
            mask_name = f"{base_name}_mask.png"
            mask_path = (PROCESSED_BRACKET_BLACK_DATA_DIR / "ground_truth" / "scratches" / mask_name)
            if mask_path.exists():
                shutil.copy(mask_path, val_mask_dir / mask_name)
            else:
                print(f"Warning: Mask not found for {img_name}")
    
    num_val_images = len(list(val_img_dir.glob("*.png")))
    num_val_masks = len(list(val_mask_dir.glob("*.png")))
    print(f"Total number of val images: {num_val_images}")
    print(f"Total number of val masks: {num_val_masks}")

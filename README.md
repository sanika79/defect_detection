# 🖼️ Computer Vision Project: Defect Detection  

This project implements **defect detection in mechanical components** using a **U-Net segmentation model**.  

It covers:  
- Environment setup with Torch + CUDA support  
- Dataset preprocessing  
- Model training  
- Inference  

---

## ⚙️ Environment Setup  

This project was tested with the following environment:  

- **Python**: 3.12.10  
- **Torch**: 2.2.2  
- **CUDA**: 11.8  
- **Torchvision**: 0.17.2  

### Steps to Set Up  
1. Clone the repository:  
   ```bash
   git clone https://github.com/sanika79/defect_detection.git
   cd defect_detection

```bash
pyenv local 3.12.10```bash
Install dependencies with Poetry:

bash
Copy code
poetry install
To add any new package:

bash
Copy code
poetry add <package-name>
Update paths in config/dev.yaml for raw and processed data. (Paths are kept as generic as possible.)

## Dataset Preprocessing
Out of the six provided mechanical components, this project focuses on bracket_black.

Original dataset structure:

bash
Copy code
bracket_black/
│── train/
│    └── good/
│── test/
│    ├── good/
│    ├── hole/
│    └── scratches/
│── ground_truth/
     ├── hole/
     └── scratches/

Preprocessing Steps
Defective test images (test/hole and test/scratches) have binary masks in ground_truth/hole and ground_truth/scratches.

Good images have no defects, so they are paired with an empty black mask.

Images and masks are renamed with a consistent convention:

Copy code
000_good.png      → 000_good_mask.png
007_hole.png      → 007_hole_mask.png
010_scratch.png   → 010_scratch_mask.png
Good and defect images are merged and split into train/validation sets using stratified sampling:

70% good + 70% defect → train

30% good + 30% defect → val

Final processed dataset structure:

kotlin
Copy code
processed_dataset/
│── train/
│    ├── images/
│    └── masks/
│── val/
     ├── images/
     └── masks/
Dataset Scripts
prepare_dataset.py → Creates processed train/val splits.

img_mask_matching.py → Verifies correct image-to-mask mappings.

 Training
Once preprocessing is complete, train the U-Net model:

bash
Copy code
python u_net_train.py
Checkpoints are saved under the lightning_logs/ directory.

Use the saved checkpoints for inference.

Inference
Run inference with:

bash
Copy code
python u_net_infer.py --checkpoint <path_to_checkpoint>
This will generate predictions (defect masks) for the given input images.

Summary
Dataset preparation ensures correct mapping of images and masks.

U-Net is trained with stratified splits of good and defect images.

Checkpoints are stored for reproducibility.

Inference can be performed using trained models.

yaml
Copy code

---

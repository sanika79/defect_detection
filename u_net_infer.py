import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import pytorch_lightning
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from u_net_train import LitUNet  

VAL_IMG_DIR = "data/processed/bracket_black/dataset/val/images"
VAL_MASK_DIR = "data/processed/bracket_black/dataset/val/masks"
CKPT_PATH = "lightning_logs/test_6/checkpoints/last.ckpt"  
RESULTS_DIR = "results/val_predictions"
W, H = 384, 384
os.makedirs(RESULTS_DIR, exist_ok=True)


class ValDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.img_dir.glob("*.png")))
        self.masks = sorted(list(self.mask_dir.glob("*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((W, H)),  # match your training size
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask, self.images[idx].name



model = LitUNet.load_from_checkpoint(CKPT_PATH)
model.eval()
model.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

val_ds = ValDataset(VAL_IMG_DIR, VAL_MASK_DIR)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# Inference 
dice_scores = []
accuracies = []
for imgs, masks, names in val_loader:
    imgs = imgs.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))
        preds_bin = (preds > 0.5).float()
    # Save prediction
    pred_np = preds_bin.squeeze().cpu().numpy() * 255
    pred_img = Image.fromarray(pred_np.astype(np.uint8))
    pred_img.save(os.path.join(RESULTS_DIR, f"pred_{names[0]}"))
    # Compute Dice
    mask_bin = (masks > 0.5).float()   ## 
    intersection = (preds_bin * mask_bin).sum().item()
    dice = (2. * intersection) / (preds_bin.sum().item() + mask_bin.sum().item() + 1e-7)
    dice_scores.append(dice)
    # Compute Accuracy
    correct = (preds_bin == mask_bin).sum().item()  ## correct pixels
    total = mask_bin.numel()   ## total number of pixels in ground truth mask tensor
    acc = correct / total   ## accuracy = correct pixels / total pixels 
    accuracies.append(acc)
    print(f"{names[0]}: Dice={dice:.4f}, Accuracy={acc:.4f}")

    #  Visualization 
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(imgs.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[1].imshow(masks.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')
    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title(f'Prediction\nDice_score={dice:.4f}')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

print(f"Mean Dice on validation set: {np.mean(dice_scores):.4f}")
print(f"Mean Accuracy on validation set: {np.mean(accuracies):.4f}")
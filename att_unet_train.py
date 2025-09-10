import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from torchvision import transforms

EPOCHS = 150
POS_WEIGHT = 3.0
LR = 1e-3
W,H = 384,384  # Image dimensions


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.image_dir.glob("*.png")))
        self.masks = sorted(list(self.mask_dir.glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ])
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = self.transform(img)
        mask = self.transform(mask)
        mask = (mask > 0).float()
        return img, mask

#  Attention Block 
## This block computes an attention map to focus the model on relevant spatial regions during upsampling.
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder after upsample)   shape [B, F_g, H, W]
        # x: skip connection (from encoder)                shape [B, F_l, H, W]
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)            # shape [B,1,H,W]
        return x * psi                 # apply attention map to skip

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.downs = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder - started from bottleneck channels (features[-1]*2)
        self.up_transposes = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        in_ch_up = features[-1]*2  # 1024 for features=[64,128,256,512]
        for feat in reversed(features):
            # upsample from current in_ch_up to feat channels
            self.up_transposes.append(nn.ConvTranspose2d(in_ch_up, feat, kernel_size=2, stride=2))
            # attention block: gating F_g = feat, skip F_l = feat
            self.att_blocks.append(AttentionBlock(F_g=feat, F_l=feat, F_int=feat//2))
            # after concat (feat from up + feat from skip) to feat*2 channels into DoubleConv
            self.up_convs.append(DoubleConv(feat*2, feat))
            # next iteration in_ch_up is feat (after conv)
            in_ch_up = feat

        # final conv
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # reverse skips for decoder
        skips = skips[::-1]

        # Decoder
        for i in range(len(self.up_transposes)):
            x = self.up_transposes[i](x)          # upsample
            skip = skips[i]                       # corresponding skip map

            # pad x if needed to match skip spatial dims
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = nn.functional.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])

            # attention gating: gating uses x (upsampled decoder) and skip
            attn = self.att_blocks[i](g=x, x=skip)  # returns skip * attention_map
            # concat attended skip + upsampled decoder feature
            x = torch.cat([attn, x], dim=1)
            # conv to reduce channels
            x = self.up_convs[i](x)

        x = self.final_conv(x)
        return x

# Losses 
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        return 1 - dice_score

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=POS_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)

def dice_coeff(preds, targets, eps=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)

# Lightning Module 
class LitAttUNet(pl.LightningModule):
    def __init__(self, lr=LR, pos_weight=POS_WEIGHT):
        super().__init__()
        self.model = AttUNet()
        self.loss_fn = BCEDiceLoss(pos_weight=pos_weight)
        self.lr = lr
        self.train_dice_losses = []
        self.train_bce_losses = []
        self.val_dice_losses = []
        self.val_bce_losses = []
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        bce_loss = self.loss_fn.bce(logits, masks)
        dice_loss = self.loss_fn.dice(logits, masks)
        loss = bce_loss + dice_loss
        self.train_bce_losses.append(bce_loss.item())
        self.train_dice_losses.append(dice_loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        bce_loss = self.loss_fn.bce(logits, masks)
        dice_loss = self.loss_fn.dice(logits, masks)
        loss = bce_loss + dice_loss
        preds = torch.sigmoid(logits)
        preds_bin = (preds > 0.5).float()
        dice = dice_coeff(preds_bin, masks)
        self.val_bce_losses.append(bce_loss.item())
        self.val_dice_losses.append(dice_loss.item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice_score", dice, prog_bar=True)
        return {"imgs": imgs, "masks": masks, "preds": preds_bin}
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

#  Training 
def main():
    data_root = Path("data/processed/bracket_black/dataset")
    train_ds = SegmentationDataset(data_root/"train/images", data_root/"train/masks")
    val_ds = SegmentationDataset(data_root/"val/images", data_root/"val/masks")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    model = LitAttUNet(lr=LR, pos_weight=POS_WEIGHT)
    checkpoint = ModelCheckpoint(
        monitor="val_dice_score",
        save_top_k=1,
        mode="max",
        save_last=True
    )
    early_stop = EarlyStopping(monitor="val_dice_score", patience=130, mode="max", verbose=True)
    trainer = pl.Trainer(
        max_epochs= EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint, early_stop]
    )
    trainer.fit(model, train_loader, val_loader)
    # Plot BCE and Dice loss curves after training
    epochs = range(1, len(model.train_bce_losses) // len(train_loader) + 1)
    def per_epoch_means(losses, loader):
        n = len(loader)
        return [np.mean(losses[i*n:(i+1)*n]) for i in range(len(losses)//n)]
    train_bce = per_epoch_means(model.train_bce_losses, train_loader)
    train_dice = per_epoch_means(model.train_dice_losses, train_loader)
    val_bce = per_epoch_means(model.val_bce_losses, val_loader)
    val_dice = per_epoch_means(model.val_dice_losses, val_loader)
    save_dir = "results/plots_dice_loss"
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(epochs, train_bce, label='Train BCE Loss')
    plt.plot(epochs, val_bce, label='Val BCE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.title('BCE Loss per Epoch')
    plt.savefig(os.path.join(save_dir, 'bce_loss_curve.png'))
    plt.close()
    plt.figure()
    plt.plot(epochs, train_dice, label='Train Dice Loss')
    plt.plot(epochs, val_dice, label='Val Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.legend()
    plt.title('Dice Loss per Epoch')
    plt.savefig(os.path.join(save_dir, 'dice_loss_curve.png'))
    plt.close()

if __name__ == "__main__":
    main()

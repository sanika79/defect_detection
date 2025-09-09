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
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchvision import transforms

# -----------------------
# Dataset
# -----------------------
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
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        img = self.transform(img)  # shape: [3, 256, 256]
        mask = self.transform(mask)  # shape: [1, 256, 256], float in [0,1]

        # Binarize mask (if needed)
        mask = (mask > 0).float()

        return img, mask


# -----------------------
# U-Net Model
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = T.Resize(skip_connection.shape[2:])(x)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# -----------------------
# Metrics
# -----------------------
def dice_coeff(preds, targets, eps=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)


def iou_score(preds, targets, eps=1e-7):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


# -----------------------
# Lightning Module
# -----------------------
class LitUNet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = UNet()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)

        preds = torch.sigmoid(logits)
        preds_bin = (preds > 0.5).float()

        dice = dice_coeff(preds_bin, masks)
        iou = iou_score(preds_bin, masks)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)
        return {"imgs": imgs, "masks": masks, "preds": preds_bin}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


# -----------------------
# Visualization Callback
# -----------------------
class VisualizePredictionsCallback(Callback):
    def __init__(self, num_samples=2):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = trainer.callback_metrics
        data_root = Path("data/processed/bracket_black/dataset")
        val_ds = SegmentationDataset(data_root/"val/images", data_root/"val/masks")
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)


        batch = next(iter(val_loader))
        imgs, masks = batch
        imgs, masks = imgs.to(pl_module.device), masks.to(pl_module.device)
        preds = torch.sigmoid(pl_module(imgs))
        preds_bin = (preds > 0.3).float()

        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy().squeeze(1)
        preds_bin = preds_bin.cpu().numpy().squeeze(1)

        import os
        epoch = trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0
        save_dir = os.path.join("plots", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(self.num_samples, len(imgs))):
            fig, axs = plt.subplots(1, 3, figsize=(10, 4))
            axs[0].set_title("Image")
            axs[0].imshow(imgs[i])
            axs[1].set_title("Ground Truth")
            axs[1].imshow(masks[i], cmap="gray")
            axs[2].set_title("Prediction")
            axs[2].imshow(preds_bin[i], cmap="gray")
            for ax in axs: ax.axis("off")
            plot_path = os.path.join(save_dir, f"plot_{i}.png")
            plt.savefig(plot_path)
            plt.close(fig)


# -----------------------
# Training
# -----------------------
def main():
    data_root = Path("data/processed/bracket_black/dataset")

    train_ds = SegmentationDataset(data_root/"train/images", data_root/"train/masks")
    val_ds = SegmentationDataset(data_root/"val/images", data_root/"val/masks")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    model = LitUNet(lr=1e-3)

    checkpoint = ModelCheckpoint(monitor="val_dice", save_top_k=1, mode="max")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint, VisualizePredictionsCallback(num_samples=2)]
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

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
        mask = Image.open(mask_path).convert("L")  # grayscale

        img = self.transform(img)  # shape: [3, W, W]
        mask = self.transform(mask)  # shape: [1, W, W], float in [0,1]

        # Binarize mask (if needed)
        mask = (mask > 0).float()

        return img, mask



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
        #print(f"UNet Initial input shape: {x.shape}")
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            #print(f"UNet Down block output shape: {x.shape}")
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        #print(f"UNet Bottleneck shape: {x.shape}")
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            ## Aligns shapes by padding if necessary
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            # if x.shape != skip_connection.shape:
            #     x = T.Resize(skip_connection.shape[2:])(x)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)
            #print(f"UNet Up block {idx // 2} output shape: {x.shape}")

        #print("self.final_conv(x).shape:", self.final_conv(x).shape)

        return self.final_conv(x)



class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)   # convert logits → [0,1]
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        return 1 - dice_score


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=POS_WEIGHT):
        super().__init__()
        # weight positives more since defects are rare
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)


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


class LitUNet(pl.LightningModule):
    def __init__(self, lr=LR, pos_weight=POS_WEIGHT):
        super().__init__()
        self.model = UNet()
        self.loss_fn = BCEDiceLoss(pos_weight=pos_weight)
        self.lr = lr
        self.train_dice_losses = []
        self.train_bce_losses = []
        self.val_dice_losses = []
        self.val_bce_losses = []

    def forward(self, x):
        ## print("model output tensor shape:", self.model(x).shape)  ## Debugging line
        return self.model(x)  ## torch.Size([B, 1, W, W])

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

        preds = torch.sigmoid(logits)   # torch.Size([B, 1, W, W])
        preds_bin = (preds > 0.5).float()

        dice = dice_coeff(preds_bin, masks)
        iou = iou_score(preds_bin, masks)

        self.val_bce_losses.append(bce_loss.item())
        self.val_dice_losses.append(dice_loss.item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice_score", dice, prog_bar=True)
        self.log("val_iou_score", iou, prog_bar=True)
        return {"imgs": imgs, "masks": masks, "preds": preds_bin}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)



class VisualizePredictionsCallback(Callback):
    def __init__(self, num_samples=4):
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
        save_dir = os.path.join("results/plots_dice_loss", f"epoch_{epoch}")
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



def main():
    data_root = Path("data/processed/bracket_black/dataset")

    train_ds = SegmentationDataset(data_root/"train/images", data_root/"train/masks")
    val_ds = SegmentationDataset(data_root/"val/images", data_root/"val/masks")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    model = LitUNet(lr=LR, pos_weight=POS_WEIGHT) 


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
        callbacks=[checkpoint, early_stop, VisualizePredictionsCallback(num_samples=4)]
    )

    trainer.fit(model, train_loader, val_loader)

    #Plot BCE and Dice loss curves after training
    import matplotlib.pyplot as plt
    epochs = range(1, len(model.train_bce_losses) // len(train_loader) + 1)
    # Reshape to per-epoch means
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

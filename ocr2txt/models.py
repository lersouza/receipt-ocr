import string
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from transformers import T5ForConditionalGeneration, T5Model


class DoubleConvBlock(nn.Module):
    """ Adapted from: https://github.com/hiepph/unet-lightning/blob/master/Unet.py """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        return self.block(image)


class UNetDownBlock(nn.Module):
    """ Adapted from: https://github.com/hiepph/unet-lightning/blob/master/Unet.py """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, image):
        return self.block(image)


class UNetUpBlock(nn.Module):
    """ Extracted from: https://github.com/hiepph/unet-lightning/blob/master/Unet.py """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1) ## why 1?

        return self.conv(x)


class UNet(nn.Module):
    """ Extracted from: https://github.com/hiepph/unet-lightning/blob/master/Unet.py """

    def __init__(self, in_channels: int, out_classes: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.bilinear = True

        # UNet Arch
        self.inc = DoubleConvBlock(self.in_channels, 64)

        # ---> Down
        self.down1 = UNetDownBlock(64, 128)
        self.down2 = UNetDownBlock(128, 256)
        self.down3 = UNetDownBlock(256, 512)
        self.down4 = UNetDownBlock(512, 512)


        # ---> Up
        self.up1 = UNetUpBlock(1024, 256)
        self.up2 = UNetUpBlock(512, 128)
        self.up3 = UNetUpBlock(256, 64)
        self.up4 = UNetUpBlock(128, 64)

        # ---> Classifier
        self.out = nn.Conv2d(64, self.out_classes, kernel_size=1)

    def forward(self, image):
        x1 = self.inc(image)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)


class EncoderPreTrainingModel(pl.LightningModule):
    """ Checkout https://github.com/hiepph/unet-lightning/blob/master/Unet.py"""
    def __init__(self, image_width: int, image_height: int, seq_length: int,
                 t5_model: str, learning_rate: float):
        super().__init__()

        self.save_hyperparameters()

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.d_model = self.t5.config.d_model

        self.encoder_classes = len(string.digits + string.ascii_letters) + 1  # Account for UNK
        self.encoder = UNet(3, self.encoder_classes)
        
        self.encoder_expand = nn.Conv2d(self.encoder_classes, 128, kernel_size=1)

        self.adapter = nn.Sequential(
            nn.Linear(image_width * image_height, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, self.d_model))

    def forward(self, image):
        """
        Runs the model for the given image batch.
        
        `image` is of shape: (B, C, H, W):
        - B: Batch Size
        - C: Channels (RGB=3)
        - H: Image Heights
        - W: Image Widths.
        
        This returns 2 outputs:
        - `[0]`: The logits after passing `image` throught U-Net. Shape: (B, Co, H, W)
        - `[1]`: The logits after passing `[0]` through Embedding adapter. Shape: (B, W, D)

        Co: The number of categories in the segmentation Task (#chars + #digits)
        L: Sequence Length to be considered in this model (hparams.seq_length).
        """
        batch_size = image.shape[0]

        encoded = self.encoder(image)  # Shape: (B, Co, H, W)
        
        expanded = self.encoder_expand(encoded)  # Shape: (B, 128, H, W)
        flat = expanded.view(batch_size, 128, -1)  # Shape: (B, Co, H, W)

        adapted = self.adapter(flat)  # Shape: (B, 128, self.d_model)

        return encoded, adapted

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        image_logits, adapted_emb = self(batch['image'])
        image_expected = batch['mask']

        emb_expected = self.t5.get_input_embeddings()(batch['target'])  # Shape: (B, L, D)
        emb_expected = torch.mean(emb_expected, dim=1)  # Shape: (B, D)

        adapted_emb = torch.mean(adapted_emb, dim=1) # Shape: (B, D)

        loss1 = F.cross_entropy(image_logits, image_expected)
        loss2 = F.cosine_embedding_loss(adapted_emb, emb_expected, torch.ones(adapted_emb.shape[0]))

        self.log('train_segmentation_loss', loss1, on_step=True)
        self.log('train_embedding_loss', loss2, on_step=True)

        loss = loss1 + loss2

        self.log('train_loss', loss, on_step=True)

        return loss


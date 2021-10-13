from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import iou
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import segmentation_models_pytorch as smp
import albumentations as album
import rasterio

from metrics import intersection_and_union
from loss_functions import XEDiceLoss

from dataset import FloodDataset

# These transformations will be passed to our model class
training_transformations = album.Compose(
    [
     album.RandomCrop(256, 256),
     album.RandomRotate90(),
     album.HorizontalFlip(),
     album.VerticalFlip(),
    ]
)


class FloodModel(pl.LightningModule):
    def __init__(self, hparams):
        super(FloodModel, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.architecture = self.hparams.get("architecture", "Unet")
        print(self.architecture)
        self.backbone = self.hparams.get("backbone", "resnet34")
        self.weights = self.hparams.get("weights", "imagenet")
        self.lr = self.hparams.get("lr", 1e-3)
        self.max_epochs = self.hparams.get("max_epochs", 30)
        self.min_epochs = self.hparams.get("min_epochs", 6)
        self.patience = self.hparams.get("patience", 4)
        self.num_workers = self.hparams.get("num_workers", 2)
        print(self.num_workers)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.x_train = self.hparams.get("x_train")
        self.y_train = self.hparams.get("y_train")
        self.x_val = self.hparams.get("x_val")
        self.y_val = self.hparams.get("y_val")
        self.output_path = self.hparams.get("output_path", "model-outputs")
        self.gpus = self.hparams.get("gpus", False)
        print(self.gpus)
        self.transform = training_transformations

        # Where final model will be saved
        self.output_path = Path.cwd() / self.output_path
        self.output_path.mkdir(exist_ok=True)

        # Track validation IOU globally (reset each epoch)
        self.intersection = 0
        self.union = 0

        # Instantiate datasets, model, and trainer params
        self.train_dataset = FloodDataset(
            self.x_train, self.y_train, transforms=self.transform
        )
        self.val_dataset = FloodDataset(self.x_val, self.y_val, transforms=None)
        self.model = self._prepare_model()
        self.trainer_params = self._get_trainer_params()

    # Required LightningModule methods

    def forward(self, image):
        # Forward pass through the network
        return self.model(image)

    def training_step(self, batch, batch_idx):
        # Swtich on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpus:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        
        # Forward pass
        preds = self.forward(x)
        
#         print('training_step checking preds and y')
#         print(preds)
#         print(type(preds))
#         print(y)
#         print(type(y))
        
        # Calculate training loss
        criterion = XEDiceLoss()
        xe_dice_loss = criterion(preds, y)
        print('Successfully calculated loss.')
        
        # Log batch xe_dice_loss
        self.log(
            "xe_dice_loss",
            xe_dice_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return xe_dice_loss

    def validation_step(self, batch, batch_idx):
        # Switch on evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpus:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass & softmax
        preds = self.forward(x)
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1

        # Calculate validation IOU (global)
        intersection, union = intersection_and_union(preds, y)
        self.intersection += intersection
        self.union += union
        
        # Log batch IOU
        batch_iou = intersection / union
#         For newer pl versions:
        self.log(
            "iou", batch_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return batch_iou

    def train_dataloader(self):
        # DataLoader class for training
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # DataLoader class for training
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Define Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=self.patience
        )

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "val_iou",
        } # logged value to monitor
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        # Calculate IOU at the end of epoch
        intersection = self.intersection
        union = self.union
        epoch_iou = intersection / union

        # Reset metrics before next epoch
        self.intersection = 0
        self.union = 0

        # Log epoch validation IOU
        self.log("val_iou", epoch_iou, on_epoch=True, prog_bar=True, logger=True)
        return epoch_iou

    ## Convenience Methods ##

    def _prepare_model(self):
        cls = getattr(smp, self.architecture)
        model = cls(
           encoder_name=self.backbone,
           encoder_weights=self.weights,
           in_channels=2,
           classes=2,
        )
        if self.gpus:
            model.cuda()
        return model

    def _get_trainer_params(self):
        # Define callback behavior
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_iou",
            mode="max",
            verbose=True,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_iou",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )
        
        # Removing for now since it causes a recursion error
#         multiplicative = lambda epoch: 1.1
#         backbone_finetuning = BackboneFinetuning(5, multiplicative)

        # Specify where Tensorboard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard-logs")
        self.log_path.mkdir(exist_ok=True)
        logger = TensorBoardLogger(self.log_path, name="resnet-model")
#         wandb_logger = WandbLogger(project="Driven-Data-Floodwater-Mapping", entity="effective-altruism-techs")
        
        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            "gpus": 1,
            "fast_dev_run": self.hparams.get("fast_dev_run", False),
            "num_sanity_val_steps": self.hparams.get("val_sanity_checks", 0),
        }
        return trainer_params

    def fit(self):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)
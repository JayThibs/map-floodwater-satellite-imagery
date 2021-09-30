import numpy as np
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import os

class FloodModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.architecture = os.environ('MODEL_ARCHITECTURE')
        self.backbone = self.hparams.get("backbone", "resnet34")
        cls = getattr(smp, self.architecture)
        self.model = cls(
           encoder_name=self.backbone,
           encoder_weights=None,
           in_channels=2,
           classes=2,
        )

    def forward(self, image):
        # Forward pass
        return self.model(image)

    def predict(self, x_arr):
        # Switch on evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Perform inference
        preds = self.forward(torch.from_numpy(x_arr))
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1
        return preds.detach().numpy().squeeze().squeeze()
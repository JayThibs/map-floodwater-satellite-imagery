import numpy as np
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import os
from io import BytesIO

pl.seed_everything(3407)

class FloodModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("Instantiating model...")
        self.architecture = 'DeepLab' # os.environ['MODEL_ARCHITECTURE']
        self.backbone = self.hparams.get("backbone", "resnet34")
        cls = getattr(smp, self.architecture)
        self.model = cls(
           encoder_name=self.backbone,
           encoder_weights=None,
           in_channels=2,
           classes=2,
        )
        print("Model instantiated.")

    def forward(self, image):
        # Forward pass
        print("Forward pass...")
        return self.model(image)

    def predict(self, x_arr):
        print("Predicting...")
        # Switch on evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)
        
        # saved with np.save and loading with np.load to preserve shape
        load_bytes = BytesIO(x_arr) # the numpy array was saved as bytes before sending it for inference
        x_arr = np.load(load_bytes, allow_pickle=True) # loading data back into the ndarray format

        # Perform inference
        preds = self.forward(torch.from_numpy(x_arr))
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1
        print("Finished performing inference.")
        return preds.detach().numpy().squeeze().squeeze()
import numpy as np
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import os

pl.seed_everything(9)

class FloodModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("Instantiating model...")
        # Please change the architecture to DeepLabV3 if the top model was DeepLabV3
        # As stated in the notebook, I haven't been able to figure out how to pass
        # environment variables to the deployed endpoint yet.
        # I reached out to Amazon, but have not gotten a response as of yet.
        self.architecture = 'Unet' # os.environ['SM_MODEL_ARCHITECTURE']
        self.backbone = "resnet34"
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
        
        # Data sent to endpoint
#         print(x_arr)
#         print(type(x_arr))

        # Perform inference
        preds = self.forward(torch.from_numpy(x_arr))
        preds = torch.softmax(preds, dim=1)[:, 1]
        print(preds)
        preds = (preds > 0.5) * 1
        print("Finished performing inference.")
        return preds.detach().numpy().squeeze().squeeze()
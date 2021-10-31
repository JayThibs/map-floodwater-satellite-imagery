"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union, List, Optional
import torch
import pandas as pd
import numpy as np
import rasterio
import albumentations as A


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transforms
        function that takes a datum and returns the same
    """

    def __init__(
        self,
        x_paths: pd.DataFrame,
        y_paths: pd.DataFrame,
        transforms: A.Compose = None,
    ) -> None:
        super().__init__()
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray, Optional[np.ndarray]]:
        """
        Loads both VV and VH images, applies normalization and transformations, and returns a dictionary.

        Parameters
        ----------
        idx

        Returns
        -------
        sample
        """
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        with rasterio.open(img.vv_path) as vv:
            vv_path = vv.read(1)
        with rasterio.open(img.vh_path) as vh:
            vh_path = vh.read(1)
        x_arr = np.stack([vv_path, vh_path], axis=-1)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        x_arr = np.clip(x_arr, min_norm, max_norm)
        x_arr = (x_arr - min_norm) / (max_norm - min_norm)

        # Apply data augmentation, if provided
        if self.transforms:
            x_arr = self.transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1])

        # Prepare sample dictionary
        sample = {"chip_id": img.chip_id, "chip": x_arr}

        # Load label if available - training only
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)
            # Apply same data augmentation to label
            if self.transforms:
                y_arr = self.transforms(image=y_arr)["image"]
            sample["label"] = y_arr

        return sample

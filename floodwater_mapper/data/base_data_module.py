"""Base DataModule class."""
from pathlib import Path
import os
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl
import albumentations as A

import rasterio
import pandas_path

from floodwater_mapper.data.util import BaseDataset
from floodwater_mapper import util


def load_and_print_info(data_module_class) -> None:
    """Load Sentinel-1 DrivenData data and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    """
    Download raw dataset and place in data/downloaded subdirectory.

    Make sure to create a metadata.toml file in data/raw/{data_dirname}
    with the download url, SHA-256, and filename of the downloaded file.
    You will need one for each dataset and you will need to load that
    specific metadata.toml file in each specific data module.
    """
    dl_dirname.mkdir(
        parents=True, exist_ok=True
    )  # create directory in root data/downloaded if it doesn't exist
    filename = (
        dl_dirname / metadata["filename"].split(".")[0]
    )  # remove zip extension, since we delete the zip after extraction. We only check if data_dir exists.
    if filename.exists():
        return filename  # no need to run the rest if we already downloaded data
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that listed in metadata document."
        )
    print("Extracting...")
    util.extract_zip(filename, dl_dirname)
    os.remove(str(dl_dirname / metadata["filename"]))  # delete zip file
    return filename


# Below are the hard-coded hyperparameters for our data.
# These can be changed by adding a flag of the parameter to the training run.
BATCH_SIZE = 16  # batch size for training
NUM_WORKERS = 0  # number of workers for data loading
TRAINING_TRANSFORMATIONS = A.Compose(
    [
        A.RandomResizedCrop(512, 512, scale=(0.75, 1.0), p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(p=0.5),
    ]
)  # transformations to apply to training data


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        # Initialize the arguments passed in the pl.Trainer.from_argparse_args(args, ...)
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # checks to see if we're running on gpu or not
        self.on_gpu = isinstance(self.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.data_train: Union[
            BaseDataset, ConcatDataset
        ]  # BaseDataset is in data/util.py
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[1] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--data_transforms",
            default=TRAINING_TRANSFORMATIONS,
            help="Image data transformations during training.",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping,
        }

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        # DataLoader class for training
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.on_gpu,  # only True is using GPU, can simply put True if always GPU
        )

    def val_dataloader(self):
        # DataLoader class for training
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.on_gpu,
        )

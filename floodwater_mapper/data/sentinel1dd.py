import toml
import os
import shutil
from pathlib import Path
import pytorch_lightning as pl

from base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "sentinel1dd"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "sentinel1dd"
# PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
# PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"


class Sentinel1DD(BaseDataModule):
    """
    Data module for Sentinel-1 data from the Driven Data Competition.

    "The Sentinel-1 mission comprises two satellites performing C-band (4 to 8 GHz)
    synthetic-aperture radar (SAR) imaging, which provides an all-weather, day-and-night
    supply of images of Earth’s surface. As compared with optical sensors that detect energy
    reflected in the visible and infrared spectral bands, SAR systems operate in the microwave
    band, where long waves can penetrate through clouds, vegetation, fog, rain showers, and snow."
    From https://www.drivendata.org/competitions/81/detect-flood-water/page/385/.

    Each piece of land that the satellite's camera takes pictures of will have two images,
    one image in the VH polarization of light (vertical transmit and horizontal receive) and
    another in the VV polarization (vertical transmit and vertical receive). Both polarization
    bring out different characteristics in their images, allowing our model to learn all the
    intricacies of the land and better separate floodwater from non-floodwater.

    In order to train our model to be able to separate floodwater from non-floodwater, we also have
    label masks that go with every pair of VV and VH images. So, our goal is to train a model that
    can map the VV and VH images to the label mask images where every pixel has been annotated as
    floodwater or non-floodwater.
    """

    def __init__(self, args=None):
        super().__init__(args)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (
            1,
            *essentials["input_shape"],
        )  # Extra dimension is added by ToTensor()
        self.output_dims = (1,)

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            _essentials = json.load(f)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(
                self.x_trainval, self.y_trainval, transform=self.transform
            )
            self.data_train, self.data_val = split_dataset(
                base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42
            )

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(
                self.x_test, self.y_test, transform=self.transform
            )

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def _download_and_process_sentinel1dd():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping Sentinel-1 Driven Data dataset...")
    curdir = os.getcwd()
    os.chdir(dirname)

    # Unzip the dataset
    shutil.unpack_archive(filename, ".")

    print("Cleaning up...")
    os.remove(filename)
    os.chdir(curdir)

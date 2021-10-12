from sklearn.metrics import jaccard_score
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def process_mask(mask):
    mask_temp = mask.copy()
    mask_temp[mask == 255] = 0
    return mask_temp


def plot_preds(prediction, vv_path, vh_path, label_path):

    with rasterio.open(vv_path) as fvv:
        vv = fvv.read(1)
    with rasterio.open(vh_path) as fvh:
        vh = fvh.read(1)
    with rasterio.open(label_path) as fmask:
        mask = fmask.read(1)

    mask = process_mask(mask)

    X = np.zeros((512, 512, 2))
    X[:, :, 0] = (vh - (-17.54)) / 5.15
    X[:, :, 1] = (vv - (-10.68)) / 4.62

    _, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].imshow(X[:, :, 0])
    ax[0].set_title("vh")
    ax[1].imshow(mask)
    ax[1].set_title("gt")
    ax[2].imshow(prediction)
    ax[2].set_title("pred")
    plt.suptitle(jaccard_score(mask.flatten(), prediction.flatten()))
    plt.show()
    st.write(plt.show())

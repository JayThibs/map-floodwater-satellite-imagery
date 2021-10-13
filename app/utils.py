from sklearn.metrics import jaccard_score
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tkinter
import matplotlib

matplotlib.use("Qt5Agg")


def process_mask(mask):
    mask_temp = mask.copy()
    mask_temp[mask == 255] = 0
    return mask_temp


def plot_preds(prediction, vv, vh, mask):

    mask = process_mask(mask)

    X = np.zeros((512, 512, 2))
    X[:, :, 0] = (vh - (-17.54)) / 5.15
    X[:, :, 1] = (vv - (-10.68)) / 4.62

    with st.echo(code_location="below"):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        ax1.imshow(X[:, :, 0])
        ax1.set_title("vh")
        ax2.imshow(mask)
        ax2.set_title("gt")
        ax3.imshow(prediction)
        ax3.set_title("pred")
        plt.suptitle(jaccard_score(mask.flatten(), prediction.flatten()))
        st.pyplot(fig)

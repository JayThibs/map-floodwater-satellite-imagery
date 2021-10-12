import streamlit as st
from streamlit import caching
import numpy as np
import os
import re
import rasterio
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import JSONDeserializer
from utils import plot_preds

st.title("Floodwater Mapping with SAR Imagery")
st.markdown("***")

st.subheader("Upload the vv and vh Polarization Images as well as the ground truth.")
uploaded_files = st.file_uploader(" ", accept_multiple_files=True)
print("Uploaded file:", uploaded_files)

x_arr = None
ENDPOINT_NAME = "floodwater-tuning-211011-2225-002-49531969-2021-10-11-23-04-00"  # os.environ["ENDPOINT_NAME"]
AWS_DEFAULT_REGION = "us-east-1"  # os.environ["AWS_DEFAULT_REGION"]
print(uploaded_files[0].name)

if len(uploaded_files) == 3:

    print(uploaded_files)
    mask_regex = "[a-z]+\d+.tif"

    for file in uploaded_files:

        if file.name.endswith("vv.tif"):
            with rasterio.open(uploaded_files[0]) as vv:
                vv_img = vv.read(1)

        elif file.name.endswith("vh.tif"):
            with rasterio.open(uploaded_files[1]) as vh:
                vh_img = vh.read(1)

        elif re.match(regex, file.name):
            with rasterio.open(label_path) as fmask:
                mask = fmask.read(1)

    x_arr = np.stack([vv_img, vh_img], axis=-1)

    # Min-max normalization
    min_norm = -77
    max_norm = 26
    x_arr = np.clip(x_arr, min_norm, max_norm)
    x_arr = (x_arr - min_norm) / (max_norm - min_norm)

    # Transpose
    x_arr = np.transpose(x_arr, [2, 0, 1])
    x_arr = np.expand_dims(x_arr, axis=0)

    print(x_arr)
    print(x_arr.shape)

    st.write("The SAR Imagery is ready to be used in the model!")

    st.markdown("***")


else:
    st.write("Make sure you image is in TIF Format.")

st.subheader("Predict Images")

if x_arr is not None:
    st.write("The model is predicting the floodwater areas in the SAR images...")
    predictor = Predictor(
        ENDPOINT_NAME,
        serializer=NumpySerializer(),
        deserializer=JSONDeserializer(),
    )
    results = predictor.predict(x_arr)
    print(results)
    st.write(results)
    st.markdown("***")
    st.write(
        "Here's an RGB plot of the vh image, ground truth floodwater mask, and our prediction."
    )
    plot_preds(
        prediction=results,
        vv_path=uploaded_files[0],
        vh_path=uploaded_files[1],
        label_path=uploaded_files[2],
    )
    st.write(
        "Note: in a production setting, we would not have the ground truth. However, we're displaying it here to see how they compare."
    )


st.markdown("***")

# st.write('Try again with different inputs')

result = st.button("Try with New Images")
if result:

    uploaded_file = st.empty()
    predict_button = st.empty()
    caching.clear_cache()

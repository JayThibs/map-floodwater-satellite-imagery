import streamlit as st
import numpy as np
import time
from streamlit import caching
import rasterio

st.title("Floodwater Mapping with SAR Imagery")
st.markdown("***")

st.subheader("Upload the vv and vh Polarization Images")
uploaded_file = st.file_uploader(" ", accept_multiple_files=False)

if uploaded_file is not None:
    # Perform your Manupilations (In my Case applying Filters)
    # img = load_preprocess_image(uploaded_file)
    img = final_fun_1(uploaded_file)
    # img = load_preprocess_image(img)
    # st.write("Image Uploaded Successfully")
    st.write(img)
    img = load_preprocess_image(str(img))

    st.image(img)

else:
    st.write("Make sure you image is in TIF Format.")


st.markdown("***")

# st.write(' Try again with different inputs')

result = st.button(" Try again")
if result:

    uploaded_file = st.empty()
    predict_button = st.empty()
    caching.clear_cache()

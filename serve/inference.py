import json
import torch
import numpy as np
import os
    
from model import FloodModel


def model_fn(model_dir):
    print("Loading model...")
    model = FloodModel()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    print("Finished loading model.")
    return model


def input_fn(request_body, request_content_type):
    print("Accessing data...")
    assert request_content_type == 'application/x-npy'
    data = np.array(request_body).reshape(1, 2, 512, 512) # this should be a numpy ndarray
    print("Data has been stored.")
    return data


def predict_fn(data, model):
    print("Predicting floodwater of SAR images...")
    with torch.no_grad():
        prediction = model.predict(data)
    print("Finished prediction.")
    return prediction


def output_fn(predictions, content_type):
    print("Saving prediction for output...")
    assert content_type == 'application/x-npy'
    res = predictions.astype(np.uint8)
    print("Saved prediction, now sending data back to user.")
    return res
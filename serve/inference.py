import json
import torch
import numpy as np
import os

from model import FloodModel


def model_fn(model_dir):
    model = FloodModel()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/x-npy'
    data = request_body # this should be a numpy ndarray
    return data


def predict_fn(data, model):
    with torch.no_grad():
        prediction = model.predict(data)
    return prediction


def output_fn(predictions, content_type):
    assert content_type == 'application/x-npy'
    res = predictions.astype(np.uint8)
    return res
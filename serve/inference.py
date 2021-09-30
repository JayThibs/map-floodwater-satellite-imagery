import json
import torch
import numpy as np

from model import FloodModel


def model_fn(model_dir):
    model = FloodModel().to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    return data


def predict_fn(data, model):
    with torch.no_grad():
        prediction = model.predict(data)
    return prediction


def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.astype(np.uint8)
    return json.dumps(res)



import json
import torch

from train import FloodModel


def model_fn(model_dir):
    model = FloodModel().to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction


def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)



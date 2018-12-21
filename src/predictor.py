import torch

from argus import load_model


class Predictor:
    def __init__(self, model_path, bbox_transform, image_transform, device=None):
        self.model = load_model(model_path, device)
        self.model.nn_module.eval()

        self.bbox_transform = bbox_transform
        self.image_transform = image_transform

    def __call__(self, samples):
        input_tensors = []
        for image, bbox in samples:
            image, bbox = self.bbox_transform(image, bbox)
            tensor = self.image_transform(image)
            input_tensors.append(tensor)

        input_tensor = torch.stack(input_tensors, dim=0)
        input_tensor = input_tensor.to(self.model.device)

        with torch.no_grad():
            logits = self.model.predict(input_tensor)
            logits = logits.cpu()
            return logits

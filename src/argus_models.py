import torch

from cnn_finetune import make_model

from argus import Model
from argus.utils import to_device, detach_tensors

from src.arcface import ArcfaceModel
from src.focal_loss import FocalLoss


class CnnFinetune(Model):
    nn_module = make_model


class ArcfaceModel(Model):
    nn_module = ArcfaceModel
    loss = {
        'FocalLoss': FocalLoss
    }

    def prepare_batch(self, batch, device):
        inp, trg = batch
        return to_device(inp, device), to_device(trg, device)

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        embeddings = self.nn_module(input)
        prediction = self.nn_module.arcface(embeddings, target)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        prediction = detach_tensors(prediction)
        target = detach_tensors(target)
        embeddings = detach_tensors(embeddings)
        return {
            'prediction': self.prediction_transform(prediction),
            'embeddings': embeddings,
            'target': target,
            'loss': loss.item()
        }

    def val_step(self, batch) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target = self.prepare_batch(batch, self.device)
            embeddings = self.nn_module(input)
            prediction = self.nn_module.arcface(embeddings, target)
            loss = self.loss(prediction, target)
            return {
                'prediction': self.prediction_transform(prediction),
                'embeddings': embeddings,
                'target': target,
                'loss': loss.item()
            }

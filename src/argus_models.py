import torch

from cnn_finetune import make_model

from argus import Model
from argus.utils import to_device, detach_tensors

from src.arcface import ArcfaceModel
from src.focal_loss import FocalLoss
from src.center_loss import CenterLossModel, CenterLoss


class CnnFinetune(Model):
    nn_module = make_model


class ArcfaceModel(Model):
    nn_module = ArcfaceModel
    loss = {
        'FocalLoss': FocalLoss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss
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


class CenterLossModel(Model):
    nn_module = CenterLossModel

    def __init__(self, params):
        super().__init__(params)
        num_classes = params['nn_module']['cnn_finetune']['num_classes']
        self.center_loss = CenterLoss(feat_dim=params['nn_module']['embedding_size'],
                                      num_classes=num_classes)
        center_params = params['center_loss']
        self.optimizer_centloss = torch.optim.Adam(self.center_loss.parameters(),
                                                   lr=center_params['lr'])
        self.center_weight = center_params['weight']
        self.center_loss = self.center_loss.to(self.device)

    def prepare_batch(self, batch, device):
        inp, trg = batch
        return to_device(inp, device), to_device(trg, device)

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()
        self.optimizer_centloss.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        embeddings, prediction = self.nn_module(input)
        ce_loss = self.loss(prediction, target)
        center_loss = self.center_loss(embeddings, target) * self.center_weight
        loss = ce_loss + center_loss
        loss.backward()

        for param in self.center_loss.parameters():
            param.grad.data *= (1. / self.center_weight)

        self.optimizer.step()
        self.optimizer_centloss.step()

        prediction = detach_tensors(prediction)
        target = detach_tensors(target)
        embeddings = detach_tensors(embeddings)
        return {
            'prediction': self.prediction_transform(prediction),
            'embeddings': embeddings,
            'target': target,
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'center_loss': center_loss.item()
        }

    def val_step(self, batch) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()
        with torch.no_grad():
            input, target = self.prepare_batch(batch, self.device)
            embeddings, prediction = self.nn_module(input)
            ce_loss = self.loss(prediction, target)
            center_loss = self.center_loss(embeddings, target) * self.center_weight
            loss = ce_loss + center_loss
            return {
                'prediction': self.prediction_transform(prediction),
                'embeddings': embeddings,
                'target': target,
                'loss': loss.item(),
                'ce_loss': ce_loss.item(),
                'center_loss': center_loss.item()
            }

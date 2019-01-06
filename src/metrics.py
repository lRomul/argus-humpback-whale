import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances

from argus.engine import State
from argus.metrics.metric import Metric, METRIC_REGISTRY


def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class MAPatK(Metric):
    name = 'map_at_{k}'
    better = 'max'

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.name = self.name.format(k=k)
        METRIC_REGISTRY[self.name] = self.__class__
        self.scores = []

    def reset(self):
        self.scores = []

    def update(self, step_output: dict):
        preds = step_output['prediction'].cpu().numpy()
        trgs = step_output['target'].cpu().numpy()

        preds_idx = preds.argsort(axis=1)
        preds_idx = np.fliplr(preds_idx)[:, :self.k]

        self.scores += [apk([a], p, self.k) for a, p in zip(trgs, preds_idx)]

    def compute(self):
        return np.mean(self.scores)


class CosMAPatK(Metric):
    name = 'cos_map_at_{k}'
    better = 'max'

    def __init__(self, dataset, k=5, batch_size=32, num_workers=8):
        super().__init__()
        self.dataset = dataset
        self.data_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=False)
        self.k = k
        self.batch_size = batch_size
        self.name = self.name.format(k=k)
        METRIC_REGISTRY[self.name] = self.__class__
        self.embeddings = []
        self.class_indexes = []

    def reset(self):
        self.embeddings = []
        self.class_indexes = []

    def update(self, step_output: dict):
        preds = step_output['embeddings'].cpu().numpy()
        trgs = step_output['target'].cpu().numpy()

        self.embeddings.append(preds)
        self.class_indexes.append(trgs)

    def epoch_complete(self, state: State, name_prefix=''):
        val_embeds = np.concatenate(self.embeddings, axis=0)
        val_cls_idx = np.concatenate(self.class_indexes, axis=0)

        train_embeds = []
        train_cls_idx = []
        model = state.model
        with torch.no_grad():
            for input, target in self.data_loader:
                input = input.to(model.device)
                target = target.numpy()

                embeds = model.nn_module(input)[0].cpu().numpy()
                train_embeds.append(embeds)
                train_cls_idx.append(target)

        train_embeds = np.concatenate(train_embeds, axis=0)
        train_cls_idx = np.concatenate(train_cls_idx, axis=0)

        embeds_distance = cosine_distances(val_embeds, train_embeds)

        preds_idx = []
        for arg_pred in embeds_distance.argsort(axis=1):
            indexes = []
            for arg in arg_pred:
                index = train_cls_idx[arg]
                if index not in indexes:
                    indexes.append(index)
                if len(indexes) >= self.k:
                    break
            preds_idx.append(indexes)

        scores = [apk([a], p, self.k) for a, p in zip(val_cls_idx, preds_idx)]

        state.metrics[name_prefix + self.name] = np.mean(scores)

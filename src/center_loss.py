import torch
import torch.nn as nn

from cnn_finetune import make_model

from argus.metrics.metric import Metric
from argus.utils import AverageMeter


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class CenterLossModel(nn.Module):
    def __init__(self, cnn_finetune, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        num_classes = cnn_finetune['num_classes']
        self.num_classes = num_classes
        cnn_finetune = make_model(**cnn_finetune)
        self.features = cnn_finetune._features
        self.pool = cnn_finetune.pool
        self.dropout = cnn_finetune.dropout

        in_features = cnn_finetune._classifier.in_features
        self.fc_1 = nn.Linear(in_features, embedding_size)
        self.fc_2 = nn.Linear(embedding_size, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        emb = self.bn(x)

        prob = self.fc_2(emb)
        return emb, prob


class CenterLossMetric(Metric):
    name = 'center_loss'

    def __init__(self):
        self.center_avg_meter = AverageMeter()
        self.ce_avg_meter = AverageMeter()
        super().__init__()

    def reset(self):
        self.center_avg_meter.reset()
        self.ce_avg_meter.reset()

    def update(self, step_output: dict):
        self.center_avg_meter.update(step_output['center_loss'])
        self.ce_avg_meter.update(step_output['ce_loss'])

    def epoch_complete(self, state, name_prefix=''):
        state.metrics[name_prefix + 'center_loss'] = self.center_avg_meter.average
        state.metrics[name_prefix + 'ce_loss'] = self.ce_avg_meter.average

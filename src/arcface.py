import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from cnn_finetune import make_model


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output


class ArcfaceModel(nn.Module):
    def __init__(self, cnn_finetune, arcface, embedding_size):
        super().__init__()
        num_classes = cnn_finetune['num_classes']
        cnn_finetune = make_model(**cnn_finetune)
        self.features = cnn_finetune._features
        self.pool = cnn_finetune.pool
        self.dropout = cnn_finetune.dropout

        in_features = cnn_finetune._classifier.in_features
        self.fc_1 = nn.Linear(in_features, 1024)
        self.fc_2 = nn.Linear(1024, embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(embedding_size)

        self.arcface = ArcMarginProduct(in_features=embedding_size,
                                        out_features=num_classes,
                                        **arcface)

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_2(x)
        x = self.bn(x)
        return x

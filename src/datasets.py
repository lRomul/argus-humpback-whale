import tqdm
import time
import random
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from turbojpeg import TurboJPEG

from torch.utils.data import Dataset


def get_samples(train_val_cvs_path, train):
    data_df = pd.read_csv(train_val_cvs_path)
    if train:
        data_df = data_df[~data_df.val]
    else:
        data_df = data_df[data_df.val]
        data_df = data_df[data_df.class_index != -1]

    images = []
    class_indexes = []
    bboxes = []
    id2class_idx = dict()

    for i, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
        image = open(row.image_path, 'rb').read()
        images.append(image)
        class_indexes.append(row.class_index)
        bboxes.append((row.x0, row.y0, row.x1, row.y1))
        if row.Id in id2class_idx:
            if id2class_idx[row.Id] != row.class_index:
                raise Exception("Two different class index for one id")
        else:
            id2class_idx[row.Id] = row.class_index

    return images, class_indexes, bboxes, id2class_idx


class WhaleDataset(Dataset):
    def __init__(self, train_val_cvs_path, train,
                 bbox_transform=None,
                 image_transform=None):
        super().__init__()
        self.train_folds_path = train_val_cvs_path
        self.train = train
        self.bbox_transform = bbox_transform
        self.image_transform = image_transform
        self.turbo_jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

        self.images, self.class_indexes, self.bboxes, self.id2class_idx = \
            get_samples(train_val_cvs_path, train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        class_index = self.class_indexes[idx]
        bbox = self.bboxes[idx]

        image = self.turbo_jpeg.decode(image)

        if self.bbox_transform is not None:
            image, bbox = self.bbox_transform(image, bbox)

        if self.image_transform is not None:
            image = self.image_transform(image)

        class_index = torch.tensor(class_index)

        return image, class_index


def get_random_samples(train_val_cvs_path, train):
    data_df = pd.read_csv(train_val_cvs_path)
    if train:
        data_df = data_df[~data_df.val]
    else:
        data_df = data_df[data_df.val]
        data_df = data_df[data_df.class_index != -1]

    id2samples = defaultdict(list)
    id2count = dict()
    id2class_idx = dict()

    for i, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):
        image = open(row.image_path, 'rb').read()
        bbox = row.x0, row.y0, row.x1, row.y1

        sample = image, row.class_index, bbox
        id2samples[row.Id].append(sample)

        id2class_idx[row.Id] = row.class_index
        id2count[row.Id] = row.id_counts

    return dict(id2samples), id2count, id2class_idx


def get_balanced_probs(id2count, balance_coef=0.0):
    count_values = list(id2count.values())
    all_count = sum(count_values)

    id2prob = {i:c/all_count for i, c in id2count.items()}
    mean_prob = np.mean(list(id2prob.values()))

    id2balanced_prob = dict()

    for whale_id, prob in id2prob.items():
        prob_diff = mean_prob - prob
        prob = prob + prob_diff * balance_coef
        id2balanced_prob[whale_id] = prob
    return id2balanced_prob


class RandomWhaleDataset(Dataset):
    def __init__(self, train_val_cvs_path, train,
                 balance_coef=0,
                 size=20000,
                 bbox_transform=None,
                 image_transform=None):
        super().__init__()
        self.train_folds_path = train_val_cvs_path
        self.train = train
        self.balance_coef = balance_coef
        self.size = size
        self.bbox_transform = bbox_transform
        self.image_transform = image_transform
        self.turbo_jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')

        self.id2samples, self.id2count, self.id2class_idx = \
            get_random_samples(train_val_cvs_path, train)
        self.id2prob = get_balanced_probs(self.id2count, balance_coef)

        self.whale_id_lst = list(self.id2samples.keys())
        self.prob_lst = [self.id2prob[i] for i in self.whale_id_lst]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        seed = int(time.time() * 1000.0) + idx
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        whale_id = np.random.choice(self.whale_id_lst, p=self.prob_lst)
        image, class_index, bbox = random.choice(self.id2samples[whale_id])

        image = self.turbo_jpeg.decode(image)
        if self.bbox_transform is not None:
            image, bbox = self.bbox_transform(image, bbox)
        if self.image_transform is not None:
            image = self.image_transform(image)

        class_index = torch.tensor(class_index)
        return image, class_index

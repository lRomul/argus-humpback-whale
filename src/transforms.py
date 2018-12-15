import cv2
import torch
import random
import numpy as np

cv2.setNumThreads(0)


def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox=None):
        if bbox is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, bbox = t(image, bbox)
            return image, bbox


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class ImageBboxCrop:
    def __call__(self, image, bbox):
        image = image_crop(image, bbox)
        return image


class Scale:
    def __init__(self, size):
        self.size = tuple(size)[::-1]
        self.interpolation = cv2.INTER_AREA

    def __call__(self, image):
        image = cv2.resize(image,
                           self.size,
                           interpolation=self.interpolation)
        return image


class Grayscale:
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([image, image, image], axis=2)
        return image


class ImageToTensor:
    def __call__(self, image):
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        return image


def get_transforms(train, size):
    transforms_dict = dict()

    transforms_dict['bbox_transform'] = ImageBboxCrop()

    if train:
        image_transforms = [
            Scale(size),
            UseWithProb(Grayscale(), 0.25),
            ImageToTensor()
        ]
    else:
        image_transforms = [
            Scale(size),
            ImageToTensor()
        ]
    transforms_dict['image_transform'] = Compose(image_transforms)
    return transforms_dict

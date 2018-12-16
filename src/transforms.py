import cv2
import torch
import random
import numpy as np

cv2.setNumThreads(0)


def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def gauss_noise(image, sigma_sq):
    image = image.astype(np.uint32)
    h, w, c = image.shape
    gauss = np.random.normal(0, sigma_sq, (h, w))
    gauss = gauss.reshape(h, w)
    image = image + np.stack([gauss for _ in range(c)], axis=2)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


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


class Flip:
    def __init__(self, flip_code):
        assert flip_code == 0 or flip_code == 1
        self.flip_code = flip_code

    def __call__(self, image):
        image = cv2.flip(image, self.flip_code)
        return image


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class VerticalFlip(Flip):
    def __init__(self):
        super().__init__(0)


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, image):
        if self.sigma_sq > 0.0:
            image = gauss_noise(image,
                                np.random.uniform(0, self.sigma_sq))
        return image


class HSLColorAug:
    def __call__(self, image):
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2HSV)
        if np.random.random() < 0.1:
            added_hue = 5*np.random.randint(4)
        else:
            added_hue = np.random.randint(-10, 10)

        image[..., 0] += added_hue
        image[..., 0] = image[..., 0] % 360

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return np.uint8(image)


class YCbCrColorAug:
    def __call__(self, image):
        image = np.int16(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

        choice = np.random.choice(['blue', 'red', 'luma', 'both', 'none'])

        if np.random.random() < 0.1:
            added_cb = 10 * np.random.randint(4)
        else:
            added_cb = np.random.randint(-30, 30)
        if np.random.random() < 0.1:
            added_cr = 10 * np.random.randint(4)
        else:
            added_cr = np.random.randint(-30, 30)

        if choice == 'blue':
            image[..., 1] += added_cb
        elif choice == 'red':
            image[..., 2] += added_cr
        elif choice == 'both':
            image[..., 1] += added_cb
            image[..., 2] += added_cr
        else:
            pass
        image = np.clip(image, 0, 255)
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_YCrCb2BGR)
        return image


class RandomGaussianBlur:
    '''Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    '''
    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image):
        kernel_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernel_size, self.sigma_x)
        return blured_image


class Rotate:
    def __init__(self, n):
        self.n = n

    def __call__(self, image):
        image = np.rot90(image, k=self.n)
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
            UseWithProb(Rotate(2), 0.25),
            UseWithProb(HorizontalFlip(), 0.25),
            UseWithProb(VerticalFlip(), 0.25),
            UseWithProb(Grayscale(), 0.25),
            UseWithProb(GaussNoise(10), 0.25),
            UseWithProb(YCbCrColorAug(), 0.25),
            UseWithProb(HSLColorAug(), 0.25),
            UseWithProb(RandomGaussianBlur(), 0.25),
            ImageToTensor()
        ]
    else:
        image_transforms = [
            Scale(size),
            ImageToTensor()
        ]
    transforms_dict['image_transform'] = Compose(image_transforms)
    return transforms_dict

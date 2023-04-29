"""This file contains all the transformers and transformations used in the project."""

__all__ = [
    "training_transformation",
]


from random import random

from torchvision import transforms as T


class RandomGauusianBlur(object):
    def __init__(self, kernel_size: int = 23, p: float = 0.5):
        self.p = p
        self.gaussian_blur = T.GaussianBlur(kernel_size=kernel_size)

    def __call__(self, data):
        if random() < self.p:
            return self.gaussian_blur(data)
        return data


training_transformation = T.Compose(
    [
        T.ToTensor(),
        T.RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.08, 1.0), size=224),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.2),
        RandomGauusianBlur(kernel_size=23, p=0.5),
        T.RandomSolarize(p=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

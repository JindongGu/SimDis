import random

from PIL import ImageFilter, ImageOps
import torchvision.transforms as transforms


class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")


class Solarization(object):
    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def byol_transform(n_views=2):
    transforms_list = []
    for i in range(int(n_views/2)):
        transform_q = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        transform_k = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([Solarization()], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        transforms_list.append(transform_q)
        transforms_list.append(transform_k)
        
    return transforms_list


def typical_imagenet_transform(train):
    if train:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    return transform

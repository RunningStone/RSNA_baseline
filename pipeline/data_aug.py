
import numpy as np
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray)
from albumentations.pytorch import ToTensorV2

def get_transform(is_train=True, vertical_flip=0.5, horizontal_flip=0.5):
    if is_train:
        transform = Compose([   Resize(height=512,width=512,always_apply=True),
                                RandomResizedCrop(height=224, width=224),
                                      ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]),
                                      HorizontalFlip(p = horizontal_flip),
                                      VerticalFlip(p = vertical_flip),
                                      ToTensorV2()])
    else:
        transform = Compose([Resize(height=512,width=512,always_apply=True),
                            ToTensorV2()])
    return transform

def get_three_channels(img):
    return np.concatenate([img, img, img], axis=0)
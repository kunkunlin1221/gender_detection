from typing import Optional, Tuple, Union

import albumentations as A
import capybara as cb
import numpy as np
from capybara.enums import BORDER, INTER
from torch.utils.data import Dataset


def imresize_and_pad_if_need(
    img: np.ndarray,
    max_h: int,
    max_w: int,
    interpolation: Union[str, int, INTER] = INTER.BILINEAR,
    pad_value: Optional[Union[int, Tuple[int, int, int]]] = 0,
    pad_mode: Union[str, int, BORDER] = BORDER.CONSTANT,
    return_scale: bool = False,
):
    raw_h, raw_w = img.shape[:2]
    scale = min(max_h / raw_h, max_w / raw_w)
    dst_h, dst_w = min(int(raw_h * scale), max_h), min(int(raw_w * scale), max_w)
    img = cb.imresize(
        img,
        (dst_h, dst_w),
        interpolation=interpolation,
    )
    img_h, img_w = img.shape[:2]

    pad_w = max_w - img_w
    pad_h = max_h - img_h
    pad_size = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)
    img = cb.pad(img, pad_size, pad_value, pad_mode)
    if return_scale:
        return img, scale
    else:
        return img


class GenderDataset(Dataset):
    def __init__(self, txt_file, mode="train"):
        with open(txt_file, "r") as f:
            lines = f.readlines()
        annotations = [line.strip().split() for line in lines]
        files, labels = zip(*annotations)
        self.files = files
        self.labels = labels
        self.mother_folder = cb.Path(txt_file).parent
        self.mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        self.std = np.array([0.229, 0.224, 0.225], dtype="float32")
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0),
        ])
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= len(self.files):
            raise IndexError("Index out of bounds")
        file_path = self.mother_folder / self.files[idx]
        img = cb.imread(file_path)
        img = imresize_and_pad_if_need(img, 112, 112, pad_value=0)
        if self.mode == "train":
            img = self.transform(image=img)["image"]
        img = img.transpose(2, 0, 1)
        return img, int(self.labels[idx])

    def restore_img(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose(1, 2, 0) * self.std + self.mean
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

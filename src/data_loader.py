import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class DepthDataset(Dataset):
    def __init__(self, X_path, y_path, img_size, augment=False, aug_config=None,
                 num_cities=None):
        self.img_size = img_size
        self.augment = augment

        if augment and aug_config:
            self.flip_prob = aug_config.get("horizontal_flip_prob", 0.5)
            cj = aug_config.get("color_jitter", {})
            self.color_jitter = transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.1),
            )
            self.crop_scale = tuple(aug_config.get("random_crop_scale", [0.8, 1.0]))
        else:
            self.flip_prob = 0.0
            self.color_jitter = None
            self.crop_scale = (1.0, 1.0)

        self.images_paths = []
        self.depth_paths = []
        cities = sorted(os.listdir(X_path))
        if num_cities is not None:
            cities = cities[:num_cities]
        for city in cities:
            city_x_dir = os.path.join(X_path, city)
            if not os.path.isdir(city_x_dir):
                continue
            imgs = sorted(os.listdir(city_x_dir))
            for img in imgs:
                x_path = os.path.join(X_path, city, img)
                self.images_paths.append(x_path)
                # Depth files are .npy (preprocessed metric depth)
                name = img.rsplit("_", 1)[0] + "_disparity.npy"
                y_path_full = os.path.join(y_path, city, name)
                self.depth_paths.append(y_path_full)

    def __getitem__(self, i):
        image = Image.open(self.images_paths[i]).convert("RGB")
        depth = torch.from_numpy(np.load(self.depth_paths[i])).unsqueeze(0)  # (1, H, W)

        if self.augment:
            # Random horizontal flip (applied to both image and depth together)
            if random.random() < self.flip_prob:
                image = TF.hflip(image)
                depth = TF.hflip(depth)

            # Random resized crop (same params for both)
            if self.crop_scale[0] < 1.0:
                crop_i, crop_j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                    image, scale=self.crop_scale, ratio=(0.9, 1.1)
                )
                image = TF.resized_crop(
                    image, crop_i, crop_j, crop_h, crop_w,
                    [self.img_size, self.img_size],
                )
                depth = TF.resized_crop(
                    depth, crop_i, crop_j, crop_h, crop_w,
                    [self.img_size, self.img_size],
                    interpolation=InterpolationMode.NEAREST,
                )

            # Color jitter on image only
            if self.color_jitter is not None:
                image = self.color_jitter(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, depth

    def __len__(self):
        return len(self.images_paths)


def get_data_loader(X_dir, y_dir, img_size, batch_size, augment=False,
                    aug_config=None, num_cities=None, num_workers=4,
                    shuffle=False, pin_memory=True):
    dataset = DepthDataset(X_dir, y_dir, img_size, augment=augment,
                           aug_config=aug_config, num_cities=num_cities)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return loader

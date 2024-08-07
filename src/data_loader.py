from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
import torch


# DEFINE CUSTOM CLASS
class DepthDataset(Dataset):
    def __init__(self, X_path, y_path, img_size):
        self.img_size = img_size

        # SET TRAINING TYPE, TRANSFORMS, PREPROCESSING
        # self.transforms = transforms
        # self.preprocessing = preprocessing

        self.images_paths = []
        self.depth_paths = []
        cities = os.listdir(X_path)[:1]
        for city in cities:
            imgs = os.listdir(os.path.join(X_path, city))
            for img in imgs:
                X = os.path.join(X_path, city, img)
                self.images_paths.append(X)
                name = img.rsplit("_", 1)[0] + "_disparity.png"
                y = os.path.join(y_path, city, name)
                self.depth_paths.append(y)

    def __getitem__(self, i):

        image = Image.open(self.images_paths[i])
        depth = Image.open(self.depth_paths[i])

        # if image.mode == "L":
        # image = image.convert("RGB")

        ## APPLY TRANSFORM
        image = Compose(
            [
                Resize(
                    (self.img_size, self.img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )(image)
        depth = Compose(
            [
                Resize(
                    (self.img_size, self.img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )(depth)

        depth = torch.clamp(depth, min=0, max=22500)
        depth = depth / 100

        ## APPLY PREPROCESSING
        # if self.preprocessing:
        # image = Compose([Normalize(mean = [0.485, 0.456, 0.406],
        # std = [0.229, 0.224, 0.225])])(image)

        return image, depth

    def __len__(self):
        return len(self.images_paths)


def get_data_loader(X_dir, y_dir, img_size, batch_size):

    dataset = DepthDataset(X_dir, y_dir, img_size)

    loader = DataLoader(dataset, batch_size=batch_size)

    return loader

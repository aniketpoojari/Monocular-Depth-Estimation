import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import read_params


def resize_images(src_dir, dest_dir, img_size, num_cities=None):
    """Resize RGB images with BICUBIC interpolation."""
    if not os.path.exists(src_dir):
        return

    os.makedirs(dest_dir, exist_ok=True)
    cities = sorted(os.listdir(src_dir))
    if num_cities is not None:
        cities = cities[:num_cities]

    for city in cities:
        src_city_path = os.path.join(src_dir, city)
        dest_city_path = os.path.join(dest_dir, city)

        if not os.path.isdir(src_city_path):
            continue

        if os.path.exists(dest_city_path) and len(os.listdir(dest_city_path)) > 0:
            continue

        print(f"  Processing city: {city}")
        os.makedirs(dest_city_path, exist_ok=True)

        for img_name in tqdm(os.listdir(src_city_path), desc=city, leave=False):
            src_file = os.path.join(src_city_path, img_name)
            dest_file = os.path.join(dest_city_path, img_name)

            with Image.open(src_file) as img:
                resized = img.resize((img_size, img_size), Image.BICUBIC)
                resized.save(dest_file)


def resize_and_convert_depth(src_dir, dest_dir, img_size, baseline, focal_length,
                             max_depth, num_cities=None):
    """Resize disparity maps and convert to metric depth (meters).

    Cityscapes disparity PNGs store disparity * 256 as uint16.
    Conversion: depth = baseline * focal_length / (raw_value / 256)
    Invalid pixels (disparity == 0) are kept as 0.
    Output is saved as .npy float32 arrays with shape (H, W).
    """
    if not os.path.exists(src_dir):
        return

    os.makedirs(dest_dir, exist_ok=True)
    cities = sorted(os.listdir(src_dir))
    if num_cities is not None:
        cities = cities[:num_cities]

    for city in cities:
        src_city_path = os.path.join(src_dir, city)
        dest_city_path = os.path.join(dest_dir, city)

        if not os.path.isdir(src_city_path):
            continue

        if os.path.exists(dest_city_path) and len(os.listdir(dest_city_path)) > 0:
            continue

        print(f"  Processing city: {city}")
        os.makedirs(dest_city_path, exist_ok=True)

        for img_name in tqdm(os.listdir(src_city_path), desc=city, leave=False):
            src_file = os.path.join(src_city_path, img_name)
            base_name = os.path.splitext(img_name)[0]
            dest_file = os.path.join(dest_city_path, base_name + ".npy")

            with Image.open(src_file) as img:
                # Resize with NEAREST to preserve disparity values
                resized = img.resize((img_size, img_size), Image.NEAREST)
                raw = np.array(resized, dtype=np.float32)

                # Convert disparity to metric depth
                depth = np.zeros_like(raw)
                valid = raw > 0
                disparity = raw[valid] / 256.0  # actual disparity in pixels
                depth[valid] = (baseline * focal_length) / disparity
                depth = np.clip(depth, 0, max_depth)

                np.save(dest_file, depth)


def run_preprocessing(config_path):
    config = read_params(config_path)
    img_size = config["data"]["img_size"]
    num_cities = config["data"].get("num_cities", None)
    baseline = float(config["data"].get("baseline", 0.209313))
    focal_length = float(config["data"].get("focal_length", 2262.52))
    max_depth = float(config["data"].get("max_depth", 80.0))

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Preprocessing {split} split...")

        # Process RGB images
        src_x = os.path.join("data", "raw", "depth_X", split)
        dest_x = os.path.join("data", "processed", "depth_X", split)
        resize_images(src_x, dest_x, img_size, num_cities=num_cities)

        # Process depth maps (disparity -> metric depth in meters)
        src_y = os.path.join("data", "raw", "depth_y", split)
        dest_y = os.path.join("data", "processed", "depth_y", split)
        resize_and_convert_depth(
            src_y, dest_y, img_size, baseline, focal_length, max_depth,
            num_cities=num_cities,
        )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="params.yaml")
        run_preprocessing(parser.parse_args().config)
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user. Exiting...")
        import os
        os._exit(0)

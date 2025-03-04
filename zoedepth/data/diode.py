import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # Use a named identity function instead of a lambda
        self.normalize = self.identity
        # Resize the image to a fixed resolution (480 x 640)
        self.resize = transforms.Resize((480, 640))

        self.transform_image = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts PIL Image to (C, H, W) tensor in range [0,1]
                self.normalize,
            ]
        )
        # ToTensor converts a 2D numpy array to a tensor of shape (1, H, W)
        self.transform_depth = transforms.ToTensor()

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        image = self.transform_image(image)
        depth = self.transform_depth(depth)
        # Resize image; depth is left at its original resolution.
        image = self.resize(image)

        depth = self.resize(depth)
        return {"image": image, "depth": depth, "dataset": "diode_outdoor"}

    def identity(self, x):
        """A no-op normalization function."""
        return x


class DIODE(Dataset):
    def __init__(self, data_dir_root):
        print(f"Loading dataset from: {data_dir_root}")

        # Retrieve image file paths.
        self.image_files = glob.glob(
            os.path.join(data_dir_root, "extracted-frames", "*.png")
        )

        # Generate corresponding depth map file paths.
        self.depth_files = [
            os.path.join(
                data_dir_root,
                "depth-maps-metric",
                os.path.basename(img).replace(".png", "_depth.npy"),
            )
            for img in self.image_files
        ]

        # Ensure that depth maps exist for each image.
        valid_pairs = []
        for img, depth in zip(self.image_files, self.depth_files):
            if os.path.exists(depth):
                valid_pairs.append((img, depth))
            else:
                print(f"Warning: Missing depth map for {img}")
        if valid_pairs:
            self.image_files, self.depth_files = zip(*valid_pairs)
        else:
            self.image_files, self.depth_files = [], []
        print(f"Found {len(self.image_files)} image-depth pairs.")

        # No mask directory is used.
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        # Load image and convert to RGB.
        image = Image.open(image_path).convert("RGB")
        # Load depth as a numpy array and convert to float32 (depth values in meters).
        depth = np.load(depth_path).astype(np.float32)

        sample = {"image": image, "depth": depth}
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_diode_loader(data_dir_root, batch_size=1, num_workers=0, **kwargs):
    dataset = DIODE(data_dir_root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)


# Test the data loader
if __name__ == "__main__":
    loader = get_diode_loader(
        "/home/shakyafernando/projects/ZoeDepth/data/ground-truths/tnc",
        batch_size=1,
        # num_workers=2,
    )

    for batch in loader:
        print("Loaded a batch")
        print("Image shape:", batch["image"].shape)
        print("Depth shape:", batch["depth"].shape)
        break

# src/data/data_loader.py

import os
import rasterio
import numpy as np
from torch.utils.data import Dataset

class NRSCCloudShadowDataset(Dataset):
    """
    PyTorch Dataset for NRSC Cloud & Shadow Challenge.
    Loads TOA reflectance GeoTIFF images and corresponding label masks.
    """
    def __init__(self, images_dir, labels_dir=None, transform=None):
        """
        Args:
            images_dir (str): Directory with preprocessed TOA reflectance images (GeoTIFFs).
            labels_dir (str, optional): Directory with label masks (GeoTIFFs). If None, dataset is for inference.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
        if labels_dir:
            self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.tif')])
            assert len(self.image_files) == len(self.label_files), "Number of images and labels must match"
        else:
            self.label_files = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)  # shape: (bands, H, W)

        if self.label_files:
            label_path = os.path.join(self.labels_dir, self.label_files[idx])
            with rasterio.open(label_path) as src:
                label = src.read(1).astype(np.uint8)  # single channel mask
        else:
            label = None

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage:
if __name__ == '__main__':
    dataset = NRSCCloudShadowDataset(
        images_dir='data/processed/toa_reflectance',
        labels_dir='data/labels/manual'
    )
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Image shape: {sample["image"].shape}')
    if sample['label'] is not None:
        print(f'Label shape: {sample["label"].shape}')

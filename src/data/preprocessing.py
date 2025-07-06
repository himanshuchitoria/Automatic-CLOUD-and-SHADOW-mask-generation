# src/data/preprocessing.py

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import yaml

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def dn_to_toa_reflectance(dn, gain, bias, sun_elevation, esun, d):
    """
    Convert Digital Number (DN) to Top-of-Atmosphere (TOA) reflectance.
    Formula:
        TOA = (π * L * d^2) / (ESUN * cos(θs))
        where L = gain * DN + bias
    """
    radiance = gain * dn + bias
    theta_s = np.deg2rad(90 - sun_elevation)  # Convert elevation to zenith angle in radians
    toa = (np.pi * radiance * d ** 2) / (esun * np.cos(theta_s))
    return toa

def normalize(image, min_val=None, max_val=None):
    """Normalize image to [0, 1]."""
    if min_val is None:
        min_val = np.nanmin(image)
    if max_val is None:
        max_val = np.nanmax(image)
    return (image - min_val) / (max_val - min_val + 1e-8)

def preprocess_scene(scene_dir, metadata, out_dir, patch_size=256):
    """
    Preprocess a single LISS-4 scene:
    - Convert DN to TOA reflectance for each band
    - Normalize
    - Split into patches (optional)
    """
    band_files = {
        'green': os.path.join(scene_dir, 'BAND2.tif'),
        'red': os.path.join(scene_dir, 'BAND3.tif'),
        'nir': os.path.join(scene_dir, 'BAND4.tif')
    }
    gain = metadata['gain']  # dict: {band: value}
    bias = metadata['bias']  # dict: {band: value}
    esun = metadata['esun']  # dict: {band: value}
    sun_elevation = float(metadata['sun_elevation'])
    earth_sun_dist = float(metadata['earth_sun_distance'])

    toa_bands = []
    for band in ['green', 'red', 'nir']:
        with rasterio.open(band_files[band]) as src:
            dn = src.read(1).astype(np.float32)
            toa = dn_to_toa_reflectance(
                dn, gain[band], bias[band], sun_elevation, esun[band], earth_sun_dist
            )
            toa_norm = normalize(toa)
            toa_bands.append(toa_norm)
            profile = src.profile

    # Stack bands into a single array (C, H, W)
    stacked = np.stack(toa_bands, axis=0)

    # Save preprocessed image
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "toa_reflectance.tif")
    profile.update(count=3, dtype='float32')
    with rasterio.open(out_path, 'w', **profile) as dst:
        for i in range(3):
            dst.write(stacked[i], i+1)

    # Optional: Split into patches for training
    # (Assume H, W divisible by patch_size for simplicity)
    h, w = stacked.shape[1:]
    patch_dir = os.path.join(out_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)
    patch_id = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = stacked[:, y:y+patch_size, x:x+patch_size]
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patch_path = os.path.join(patch_dir, f"patch_{patch_id:04d}.npy")
                np.save(patch_path, patch)
                patch_id += 1

    print(f"Preprocessing complete. TOA reflectance saved to {out_path}. {patch_id} patches created.")

# Example usage:
if __name__ == "__main__":
    config = load_config("config/config.yaml")
    # Example: Load metadata from a JSON or TXT file
    # For demonstration, hardcoded values are used
    metadata = {
        'gain': {'green': 0.01, 'red': 0.01, 'nir': 0.01},
        'bias': {'green': 0.0, 'red': 0.0, 'nir': 0.0},
        'esun': {'green': 1850, 'red': 1550, 'nir': 1040},
        'sun_elevation': 45.0,
        'earth_sun_distance': 1.0
    }
    scene_dir = "data/raw/training/RAF25JAN2025042220009700055SSANSTUC00GTDB"
    out_dir = "data/processed/RAF25JAN2025042220009700055SSANSTUC00GTDB"
    preprocess_scene(scene_dir, metadata, out_dir, patch_size=256)

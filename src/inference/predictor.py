# src/inference/predictor.py

import os
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.models.attention_unet import AttentionUNet
from src.data.preprocessing import dn_to_toa_reflectance, normalize
from src.utils.shapefile_utils import save_mask_as_shapefiles
import yaml

def load_model(model_path, config, device='cpu'):
    model = AttentionUNet(
        in_channels=config['model']['input_channels'],
        out_channels=config['model']['num_classes']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_test_image(image_dir, metadata, config):
    """Preprocess test image: DN to TOA, normalization, stacking bands."""
    band_files = {
        'green': os.path.join(image_dir, 'BAND2.tif'),
        'red': os.path.join(image_dir, 'BAND3.tif'),
        'nir': os.path.join(image_dir, 'BAND4.tif')
    }
    gain = metadata['gain']
    bias = metadata['bias']
    esun = metadata['esun']
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
            transform = src.transform
            crs = src.crs

    stacked = np.stack(toa_bands, axis=0)
    return stacked, profile, transform, crs

def predict_mask(model, image, device='cpu'):
    """Run model inference on preprocessed image."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(image).unsqueeze(0).to(device)  # [1, C, H, W]
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()[0]  # [H, W]
    return preds

def save_mask_geotiff(mask, profile, out_path):
    """Save the predicted mask as 8-bit georeferenced TIFF."""
    profile.update(dtype='uint8', count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

def main(
    test_image_dir,
    metadata,
    model_path,
    config_path,
    output_dir,
    dataset_id
):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, config, device)

    # Preprocess image
    image, profile, transform, crs = preprocess_test_image(test_image_dir, metadata, config)

    # Predict mask
    mask = predict_mask(model, image, device)

    # Save georeferenced mask TIFF
    os.makedirs(output_dir, exist_ok=True)
    mask_tiff_path = os.path.join(output_dir, f"{dataset_id}_mask.tiff")
    save_mask_geotiff(mask, profile, mask_tiff_path)

    # Generate shapefiles for cloud and shadow
    cloud_shape_dir = os.path.join(output_dir, "cloudshapes")
    shadow_shape_dir = os.path.join(output_dir, "shadowshapes")
    save_mask_as_shapefiles(mask, transform, crs, cloud_shape_dir, shadow_shape_dir)

    # Zip shapefiles as required by NRSC
    import shutil
    shutil.make_archive(os.path.join(output_dir, "cloudshapes"), 'zip', cloud_shape_dir)
    shutil.make_archive(os.path.join(output_dir, "shadowshapes"), 'zip', shadow_shape_dir)

    print(f"Inference complete for {dataset_id}.")
    print(f"Mask saved: {mask_tiff_path}")
    print(f"Cloud shapefiles: {cloud_shape_dir}.zip")
    print(f"Shadow shapefiles: {shadow_shape_dir}.zip")

if __name__ == "__main__":
    # Example usage (to be replaced with actual test data paths and metadata)
    test_image_dir = "data/raw/test/RAF25JAN2025042220009700055SSANSTUC00GTDB"
    metadata = {
        'gain': {'green': 0.01, 'red': 0.01, 'nir': 0.01},
        'bias': {'green': 0.0, 'red': 0.0, 'nir': 0.0},
        'esun': {'green': 1850, 'red': 1550, 'nir': 1040},
        'sun_elevation': 45.0,
        'earth_sun_distance': 1.0
    }
    model_path = "models/best_models/unet_attention_best.pth"
    config_path = "config/config.yaml"
    output_dir = "results/predictions/test/RAF25JAN2025042220009700055SSANSTUC00GTDB"
    dataset_id = "RAF25JAN2025042220009700055SSANSTUC00GTDB"
    main(test_image_dir, metadata, model_path, config_path, output_dir, dataset_id)

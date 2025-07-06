# src/utils/shapefile_utils.py

import os
import numpy as np
from osgeo import gdal, ogr, osr

def save_mask_as_shapefiles(mask, transform, crs, cloud_dir, shadow_dir):
    """
    Convert a mask numpy array to shapefiles for cloud and shadow classes.

    Args:
        mask (np.ndarray): 2D array with values 0 (no cloud), 1 (cloud), 2 (shadow)
        transform (Affine): Affine transform of the raster (from rasterio)
        crs (CRS): Coordinate reference system of the raster (from rasterio)
        cloud_dir (str): Directory to save cloud shapefile
        shadow_dir (str): Directory to save shadow shapefile
    """
    os.makedirs(cloud_dir, exist_ok=True)
    os.makedirs(shadow_dir, exist_ok=True)

    def polygonize_class(class_value, out_dir, class_name):
        shp_path = os.path.join(out_dir, f"{class_name}.shp")
        driver = ogr.GetDriverByName('ESRI Shapefile')
        # Remove existing shapefile if it exists
        if os.path.exists(shp_path):
            driver.DeleteDataSource(shp_path)
        ds = driver.CreateDataSource(shp_path)
        # Set spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs.to_wkt())
        layer = ds.CreateLayer(class_name, srs=srs, geom_type=ogr.wkbPolygon)
        # Add a field for the class value
        field_defn = ogr.FieldDefn('class', ogr.OFTInteger)
        layer.CreateField(field_defn)
        # Prepare mask for this class
        class_mask = (mask == class_value).astype(np.uint8)
        # Create in-memory raster to polygonize
        driver_mem = gdal.GetDriverByName('MEM')
        rows, cols = class_mask.shape
        mem_raster = driver_mem.Create('', cols, rows, 1, gdal.GDT_Byte)
        # Set geotransform and projection
        mem_raster.SetGeoTransform(transform.to_gdal())
        mem_raster.SetProjection(crs.to_wkt())
        band = mem_raster.GetRasterBand(1)
        band.WriteArray(class_mask)
        band.SetNoDataValue(0)
        # Polygonize
        gdal.Polygonize(band, None, layer, 0, [], callback=None)
        # Cleanup
        ds = None
        mem_raster = None

    # Polygonize cloud class (1)
    polygonize_class(1, cloud_dir, "clouds")
    # Polygonize shadow class (2)
    polygonize_class(2, shadow_dir, "shadows")

# Example usage:
# save_mask_as_shapefiles(mask_array, transform, crs, 'output/cloudshapes', 'output/shadowshapes')

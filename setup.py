from setuptools import setup, find_packages

setup(
    name='nrsc_cloud_shadow_challenge',
    version='1.0.0',
    description='Automatic cloud and shadow mask generation for Resourcesat-2/2A LISS-4 images',
    author='Your Team Name',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'tensorflow>=2.6.0',
        'gdal>=3.2.0',
        'rasterio>=1.2.0',
        'geopandas>=0.10.0',
        'shapely>=1.7.0',
        'fiona>=1.8.0',
        'opencv-python>=4.5.0',
        'scikit-image>=0.18.0',
        'albumentations>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.0'
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)

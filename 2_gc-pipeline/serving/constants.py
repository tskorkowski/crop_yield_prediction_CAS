"""
Project level constants
"""

import numpy as np

VARIED_START = 400
VARIED_END = 200

IMAGE_BATCH = [VARIED_START, VARIED_END]

SCALE = 60
CROP = 1

# GCS
PROJECT = "supple-nature-370421"
BUCKET = "vgnn"


# Histograms
NUM_BINS = 32
MONTHS = [5,7,9]
#MONTHS = [7+1,9+1]

IMG_SOURCE_PREFIX = "images"
HIST_DEST_PREFIX = "histograms"

PIX_COUNT = 8192**2
REFLECTANCE_CONST = 10000
MAP_NAN = True
NORMALIZE = True


BANDS = {"Coastal_aerosol": False,
         "Blue": True,
         "Green": True,
         "Red": True,
         "Red_Edge_1": True,
         "Red_Edge_2": True,
         "Red_Edge_3": True,
         "NIR": True,
         "Narrow_NIR": True,
         "Water_vapor": True,
         "SWIR": False,
         "SWIR_1": False,
         "SWIR_2": False}

SELECTED_BANDS = [band for band, m in zip(range(1, len(BANDS) +1) , BANDS.values()) if m]

LABELS_PATH = "labels_combined.npy"
HEADER_PATH = "labels_header.npy"

def hist_bins(buckets: int=NUM_BINS) -> list:
    bins = [
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #1
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #2 
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #3
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #4
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #5
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #6
    np.linspace(0, 0.6 * REFLECTANCE_CONST, buckets + 1), #7
    np.linspace(0, 0.6 * REFLECTANCE_CONST, buckets + 1), #8
    np.linspace(0, 0.6 * REFLECTANCE_CONST, buckets + 1), #9
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #10
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #11
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #12
    np.linspace(0, 0.5 * REFLECTANCE_CONST, buckets + 1), #13
]
    return bins

HIST_BINS_LIST = [bins for bins, m in zip(hist_bins(),BANDS.values()) if m]

def get_bins_bands (n_buckets: int, bands: bool) -> list:
    return {"hist_bins" : [bins for bins, m in zip(hist_bins(n_buckets),bands) if m],
            "sel_bands" : [band for band, m in zip(range(1, len(BANDS) +1) , bands) if m]
           }
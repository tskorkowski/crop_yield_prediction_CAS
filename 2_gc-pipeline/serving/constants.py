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

# Sentinel 2 Params
MONTHS = [5,7,9]
IMG_SOURCE_PREFIX = "images"
HIST_DEST_PREFIX = "histograms"

PIX_COUNT = 8192**2
REFLECTANCE_CONST = 1e4

NUM_BANDS = 13


# Histograms
NUM_BINS = 128

HIST_BINS_LIST = [
    np.linspace(0, 15000, NUM_BINS + 1), #1
    np.linspace(0, 15000, NUM_BINS + 1), #2 
    np.linspace(0, 15000, NUM_BINS + 1), #3
    np.linspace(0, 15000, NUM_BINS + 1), #4
    np.linspace(0, 15000, NUM_BINS + 1), #5
    np.linspace(0, 15000, NUM_BINS + 1), #6
    np.linspace(0, 15000, NUM_BINS + 1), #7
    np.linspace(0, 15000, NUM_BINS + 1), #8
    np.linspace(0, 15000, NUM_BINS + 1), #9
    np.linspace(0, 15000, NUM_BINS + 1), #10
    np.linspace(0, 1200, NUM_BINS + 1), #11
    np.linspace(0, 15000, NUM_BINS + 1), #12
    np.linspace(0, 15000, NUM_BINS + 1), #13
]

LABELS_PATH = "labels_combined.npy"
HEADER_PATH = "labels_header.npy"

def hist_bins(buckets: int) -> list:
    return [
    np.linspace(0, 15000, buckets + 1), #1
    np.linspace(0, 15000, buckets + 1), #2 
    np.linspace(0, 15000, buckets + 1), #3
    np.linspace(0, 15000, buckets + 1), #4
    np.linspace(0, 15000, buckets + 1), #5
    np.linspace(0, 15000, buckets + 1), #6
    np.linspace(0, 15000, buckets + 1), #7
    np.linspace(0, 15000, buckets + 1), #8
    np.linspace(0, 15000, buckets + 1), #9
    np.linspace(0, 15000, buckets + 1), #10
    np.linspace(0, 1200, buckets + 1), #11
    np.linspace(0, 15000, buckets + 1), #12
    np.linspace(0, 15000, buckets + 1), #13
]
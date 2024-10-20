"""
Project level constants
"""

import numpy as np

# GCS
PROJECT = "supple-nature-370421"
BUCKET = "vgnn"
SCALE = 60
CROP = 1

# Histograms
IMG_SOURCE_PREFIX = "images"
HIST_DEST_PREFIX = "histograms"

NUM_BANDS = 13
NUM_BINS = 32

HIST_BINS_LIST = [
    np.linspace(1, 10000, NUM_BINS + 1), #1
    np.linspace(1, 10000, NUM_BINS + 1), #2 
    np.linspace(1, 10000, NUM_BINS + 1), #3
    np.linspace(1, 10000, NUM_BINS + 1), #4
    np.linspace(1, 10000, NUM_BINS + 1), #5
    np.linspace(100, 10000, NUM_BINS + 1), #6
    np.linspace(100, 10000, NUM_BINS + 1), #7
    np.linspace(100, 10000, NUM_BINS + 1), #8
    np.linspace(100, 10000, NUM_BINS + 1), #9
    np.linspace(1, 5000, NUM_BINS + 1), #10
    np.linspace(1, 150, NUM_BINS + 1), #11
    np.linspace(100, 10000, NUM_BINS + 1), #12
    np.linspace(100, 10000, NUM_BINS + 1), #13
]

LABELS_PATH = "labels_combined.npy"
HEADER_PATH = "labels_header.npy"

MONTHS = [5,7,9]
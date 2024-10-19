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
IMG_SOURCE_PREFIX = "images/"
HIST_DEST_PREFIX = "histograms"

NUM_BANDS = 13
NUM_BINS = 32

HIST_BINS_LIST = [
    np.linspace(1, 5000, NUM_BINS + 1),
    np.linspace(1, 5000, NUM_BINS + 1),
    np.linspace(1, 5000, NUM_BINS + 1),
    np.linspace(1, 5000, NUM_BINS + 1),
    np.linspace(1, 5000, NUM_BINS + 1),
    np.linspace(100, 6000, NUM_BINS + 1),
    np.linspace(100, 6000, NUM_BINS + 1),
    np.linspace(100, 6000, NUM_BINS + 1),
    np.linspace(100, 6000, NUM_BINS + 1),
    np.linspace(1, 2500, NUM_BINS + 1),
    np.linspace(1, 150, NUM_BINS + 1),
    np.linspace(100, 5500, NUM_BINS + 1),
    np.linspace(100, 5500, NUM_BINS + 1),
]

LABELS_PATH = "labels_combined.npy"
HEADER_PATH = "labels_header.npy"
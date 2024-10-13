"""
Project level constants
"""

import numpy as np

# GCS
PROJECT = "supple-nature-370421"
BUCKET = "vgnn"
SCALE = 100
CROP = 1

# Histograms
IMG_SOURCE_PREFIX = "images/"
HIST_DEST_PREFIX = "histograms/"

NUM_BANDS = 13
HIST_BINS_LIST = [
    np.linspace(1, 5000, 33),
    np.linspace(1, 5000, 33),
    np.linspace(1, 5000, 33),
    np.linspace(1, 5000, 33),
    np.linspace(1, 5000, 33),
    np.linspace(100, 6000, 33),
    np.linspace(100, 6000, 33),
    np.linspace(100, 6000, 33),
    np.linspace(100, 6000, 33),
    np.linspace(1, 2500, 33),
    np.linspace(1, 150, 33),
    np.linspace(100, 5500, 33),
    np.linspace(100, 5500, 33),
]

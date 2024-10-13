"""
Project level constants
"""

import numpy as np

# GCS
PROJECT = "supple-nature-370421"
BUCKET = "vgnn"

# Histograms
IMG_SOURCE_PREFIX = "images/"
HIST_DEST_PREFIX = "histograms/"

NUM_BANDS = 13
HIST_BINS_LIST = [
    np.linspace(1, 2500, 33),
    np.linspace(1, 2500, 33),
    np.linspace(1, 2500, 33),
    np.linspace(1, 2500, 33),
    np.linspace(1, 2500, 33),
    np.linspace(100, 5500, 33),
    np.linspace(100, 5500, 33),
    np.linspace(100, 5500, 33),
    np.linspace(100, 5500, 33),
    np.linspace(1, 1000, 33),
    np.linspace(1, 20, 33),
    np.linspace(100, 5500, 33),
    np.linspace(100, 5500, 33),
]

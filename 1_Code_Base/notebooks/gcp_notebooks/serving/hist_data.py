"""Data utilities to grab data from google cloud bucket.

Meant to be used for both training and prediction so the model is
trained on exactly the same data that will be used for predictions.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import google.auth
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from serving.constants import (
    BUCKET,
    HIST_BINS_LIST,
    HIST_DEST_PREFIX,
    IMG_SOURCE_PREFIX,
    SELECTED_BANDS,
    PROJECT,
    PIX_COUNT,
    REFLECTANCE_CONST,
    NUM_BINS
)
from google.api_core import exceptions, retry
from google.cloud import storage
from numpy.lib.recfunctions import structured_to_unstructured
from osgeo import gdal
from rasterio.io import MemoryFile
from serving.common import list_blobs_with_prefix

logging.basicConfig(
    filename="hist.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def hist_init():
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default()

def process_band(bucket, blob_name, band, bins, skip_nan, normalise):

    storage_client = storage.Client()
    blob = storage_client.bucket(bucket).blob(blob_name)

    with blob.open("rb") as f:
        with rasterio.open(f) as src:

            data = src.read(band).flatten()
            na_mask = np.isnan(data)
            
            if skip_nan == False:
                data[na_mask] = 0.0
                valid_data = data
            else:
                valid_data = data[~na_mask]
                
            if normalise:
                valid_data = valid_data / REFLECTANCE_CONST
                bins = bins / REFLECTANCE_CONST
                
            valid_max = np.max(valid_data)
            valid_min = np.min(valid_data)

            if valid_max > bins[-1]:
                logging.warning(
                    f"image: {blob_name}, band: {band}, {valid_max} value is larger than assumed possible values for this band: {bins[-1]}"
                )
            elif valid_min < bins[0]:
                logging.warning(
                    f"image: {blob_name}, band: {band}, {valid_min} value is smaller than assumed possible values for this band {bins[0]}"
                )
            
            if valid_data.size > 0:
                total_sum = np.sum(valid_data)
                total_count = valid_data.size
                mean = total_sum / total_count
                hist, _ = np.histogram(valid_data, bins=bins, density=False)
            else:
                logging.error(f"image: {blob_name}, band: {band} has 0 valid pixels. Investigate")
                mean = np.nan
                hist = np.zeros_like(
                    bins[:-1]
                )  # histogram will have one less element than bins

    return hist


def process_tiff(bucket, blob_name, bin_list, selected_bands, skip_nan, normalise, max_workers=12):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_band = {
            executor.submit(
                process_band, bucket, blob_name, band, bins, skip_nan, normalise
            ): band
            for band, bins in zip(selected_bands, bin_list)
        }
        results = []

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(f"Processed band {band} successfully")
            except Exception as exc:
                logging.exception(f"Band {band} generated an exception: {exc}")

    sorted_results = sorted(results, key=lambda x: x[0])
    return np.array(sorted_results).flatten()  # one long array instead of bands


def recombine_image(bucket, core_image_name, bin_list, selected_bands, skip_nan=False, normalise=False):
    start_time = time.time()

    hist_per_blob = []
    blobs = list_blobs_with_prefix(core_image_name)
    for blob in blobs:
        results = process_tiff(bucket, blob.name, bin_list, selected_bands, skip_nan, normalise)
        hist_per_blob.append(results)

    combined_hist = np.sum(np.array(hist_per_blob), axis=0)

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(
        f"Image {core_image_name} has been processed in {execution_time/60:.4f} minuntes"
    )
    
    return combined_hist

def write_histogram_to_gcs(histogram, bucket_name, blob_name):
    """
    Write a NumPy array (histogram) to Google Cloud Storage.

    Args:
    histogram (np.array): The histogram to save.
    bucket_name (str): The name of the GCS bucket.
    blob_name (str): The name to give the file in GCS (including any 'path').

    Returns:
    str: The public URL of the uploaded file.
    """
    # Ensure the blob_name ends with .npy
    if not blob_name.endswith('.npy'):
        blob_name += '.npy'

    # Create a GCS client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob
    blob = bucket.blob(blob_name)

    # Convert the numpy array to bytes
    array_bytes = io.BytesIO()
    np.save(array_bytes, histogram)
    array_bytes.seek(0)

    # Upload the bytes to GCS
    blob.upload_from_file(array_bytes, content_type='application/octet-stream')

    logging.info(f"Histogram uploaded to gs://{bucket_name}/{blob_name}")


if __name__ == "__main__":
    image_name = r"images/Canyon_2017_5-6_100"
    # image_name = r"images/Story_2018_9-10_100_.tif"
#     blobs = list_blobs_with_prefix(image_name)

#     # Usage
#     start_time = time.time()

#     recombine_image_hist = recombine_image(
#         BUCKET, image_name, HIST_BINS_LIST, NUM_BANDS
#     )

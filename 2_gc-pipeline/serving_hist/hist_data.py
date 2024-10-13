"""Data utilities to grab data from google cloud bucket.

Meant to be used for both training and prediction so the model is
trained on exactly the same data that will be used for predictions.
"""

from __future__ import annotations
from osgeo import gdal
import io

from google.api_core import exceptions, retry
import google.auth
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests
from typing import Dict
from google.cloud import storage
import time
import re
import tempfile
import rasterio
from rasterio.io import MemoryFile
import os
import matplotlib.pyplot as plt
from utils.constants import  BUCKET, IMG_SOURCE_PREFIX, HIST_DEST_PREFIX, NUM_BANDS, HIST_BINS_LIST
import io
from osgeo import gdal
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

def hist_init():
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default()


def list_blobs_with_prefix(bucket_name, prefix):
    """Lists all the blobs in the bucket that begin with the prefix."""
    storage_client = storage.Client()
    return storage_client.list_blobs(bucket_name, prefix=prefix)

def load_tiff_from_gcs_temp(bucket_name, blob_name):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
        temp_filename = temp_file.name

    # Download the file from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(temp_filename)

    # Read the TIFF file
    with rasterio.open(temp_filename) as src:
        # Read all bands
        tiff_array = src.read()

    # Clean up the temporary file
    os.remove(temp_filename)

    return tiff_array

def load_tiff_from_gcs_mem(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    with MemoryFile(byte_stream) as memfile:
        with memfile.open() as src:
            array = src.read()

    return array

def download_and_process_tiff(bucket_name, blob_name):
    """Downloads a TIFF blob into memory and processes it."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download the blob contents
    contents = blob.download_as_bytes()
    
    # Process the TIFF data
    with MemoryFile(contents) as memfile:
        with memfile.open() as src:
            # Read all bands
            array = src.read()
    
    return array

# Function using GDAL library directly to read image data
def read_tiff_gdal(bucket_name, blob_name):
   gdal.UseExceptions()  # Enable exceptions
   file_path = f'/vsigs/{bucket_name}/{blob_name}'
   ds = gdal.Open(file_path)
   if ds is None:
       print("Failed to open the file")
       return None
   band = ds.GetRasterBand(1)
   data = band.ReadAsArray()
   return data

def create_histogram_skip_nan(image, bins=256):
    # Flatten the image and remove NaN values
    flat_image = image.flatten()
    
    non_nan_values = flat_image[~np.isnan(flat_image)].astype(np.uint16)
    
    # Create histogram
    hist, bin_edges = np.histogram(non_nan_values, bins=bins, density=False)
    
    return hist, bin_edges

def process_band(blob_name, band, bins):

    storage_client = storage.Client()
    blob = storage_client.bucket(BUCKET).blob(blob_name)
    
    with blob.open("rb") as f:
        with rasterio.open(f) as src:    
    
            data = src.read(band)
            valid_data = data[~np.isnan(data)].astype(np.uint16)
            valid_max = np.max(valid_data)
            valid_mean = np.min(valid_data)
            
            if valid_max > bins[-1]:
                logging.warning(f"image: {image_name}, band: {band}, {valid_max} value is larger than assumed possible values for this band")
            elif valid_min < bins[0]:
                logging.warning(f"image: {image_name}, band: {band}, {valid_max} value is smaller than assumed possible values for this band")
            if valid_data.size > 0:
                total_sum = np.sum(valid_data)
                total_count = valid_data.size
                mean = total_sum / total_count
                hist, _ = create_histogram_skip_nan(valid_data, bins)
            else:
                mean = np.nan
                hist = np.zeros_like(bins[:-1])  # histogram will have one less element than bins

    return hist

def process_tiff(bucket_name, blob_name, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_band = {executor.submit(process_band, blob_name, band, HIST_BINS_LIST[band-1]): band 
                          for band in range(1, NUM_BANDS + 1)}
        results = []
        
        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(f"Processed band {band} successfully")
            except Exception as exc:
                logging.exception(f'Band {band} generated an exception: {exc}')
    
    sorted_results = sorted(results, key=lambda x: x[0])
    return  np.array(sorted_results).flatten() # one long array instead of bands

def recombine_image(blob_name):
    start_time = time.time()
    
    hist_per_blob = []
    blobs = list_blobs_with_prefix(BUCKET, blob_name)
    
    for blob in blobs:
        results = process_tiff(BUCKET, blob.name)
        hist_per_blob.append(results)
    
    combined_hist = np.sum(np.array(hist_per_blob), axis=0)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Image {blob_name} has been processed in {execution_time/60:.4f} minuntes")
    
    return combined_hist

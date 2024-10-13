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


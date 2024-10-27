# Based on:
# https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/land-cover-classification

"""Data utilities to grab data from Earth Engine.

Meant to be used for both training and prediction so the model is
trained on exactly the same data that will be used for predictions.
"""

from __future__ import annotations

import io

import ee
from google.api_core import exceptions, retry
import google.auth
from google.cloud import storage
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured
import requests
from typing import Dict
from serving.constants import SCALE, NUM_BINS, SELECTED_BANDS, HIST_DEST_PREFIX, BUCKET, LABELS_PATH, HEADER_PATH # meters per pixel, number of bins in the histogram, number of bands in the satellite image
from concurrent.futures import ThreadPoolExecutor, as_completed


def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    # Use the Earth Engine High Volume endpoint.
    #   https://developers.google.com/earth-engine/cloud/highvolume
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

def check_blob_exists(blob_name):
    # Initialize a storage client
    client = storage.Client()

    # Get the bucket object
    bucket = client.bucket(BUCKET)

    # Check if the blob exists in the bucket
    blob = bucket.blob(blob_name)

    # Verify if the blob exists
    return blob.exists()

def get_input_image_ee(county: str, county_fips: str, state_fips: str, crop: int, year: int, month: int) -> ee.Image:
    """Get a Sentinel-2 Earth Engine image.

    This filters clouds and returns the median for the selected time range and mask.
    Then it removes the mask and fills all the missing values, otherwise
    the data normalization will give infinities and not-a-number.
    Missing values on Sentinel 2 are filled with 1000, which is near the mean.
    
    Sentinel 2 image is masked with crop and bounded by county administrative borders.

    For more information, see:
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED

    Args:
        year: Year of a 

    Returns: An Earth Engine image with the median Sentinel-2 values.
    """ 
    assert month >= 1 and month <= 11, "Function provides median s2 image over two month within the same year, hence month has to be between 1 and 11"
    
    def mask_sentinel2_clouds(image: ee.Image) -> Dict:
        CLOUD_BIT = 10
        CIRRUS_CLOUD_BIT = 11
        bit_mask = (1 << CLOUD_BIT) | (1 << CIRRUS_CLOUD_BIT)
        mask = image.select("QA60").bitwiseAnd(bit_mask).eq(0)
        return image.updateMask(mask)    
   
    # Filter county
    county = county.capitalize()
    county_geom = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("COUNTYFP", county_fips))
        .filter(ee.Filter.eq("STATEFP", state_fips))
    )
    
    # Cropland data - image collection with specific crops masked
    cdl_county_masked = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterBounds(county_geom)
        .select("cropland")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .map(lambda img: img.updateMask(img.remap([crop], [1], 0)))
        .map(lambda img: img.clipToCollection(county_geom).float())
    )


    # Check if the collection is empty
    if cdl_county_masked.size().getInfo() == 0:
        print("No data found for the specified parameters.")
    else:
        # Get the first image from the filtered collection
        cdl_image = cdl_county_masked.first()

    alameda_centroid = county_geom.first().geometry().centroid()
    coords = alameda_centroid.coordinates().getInfo()

    # Define visualization parameters
    vis_params_mask = {
        'min': 0,
        'max': 1,
        'palette': ['green']
    }

    s2_img_unbounded = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filter(ee.Filter.calendarRange(year,year,"year"))
            .filter(ee.Filter.calendarRange(month,month+1,"month"))
            .filterBounds(county_geom)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
            .map(mask_sentinel2_clouds)
            .select("B.*")
            .median()
            .unmask(1000)       
            .updateMask(cdl_image.eq(1))
            .float() 
        )
    s2_img = s2_img_unbounded.clip(county_geom)
    image_name = f"{county}_{state_fips}/{year}/{month}-{month+1}"
    
    return {
            "image": s2_img,
            "image_name": image_name
    }

def get_labels(labels_path: str=LABELS_PATH, header_path: str=HEADER_PATH) -> pd.DataFrame:
    '''
    Load crop yield information into a df
    '''
    label_data = np.load(labels_path, allow_pickle=True)
    label_header = np.load(header_path, allow_pickle=True)
    label_df = pd.DataFrame(label_data, columns=label_header)
    label_df["target"] = pd.to_numeric(label_df["target"])
    
    return label_df

def get_varied_labels(count_start=0, no_records=150, ascending=False):
    """
    Create training set using counties with highest min-max spread over the years
    """
    
    labels_df = get_labels()
    
    
    df_var = labels_df.groupby(by=["county_name","state_name"]).agg(
                    count=('target', 'count'),
                    min_value=('target', 'min'),
                    max_value=('target', 'max'),
                    median=('target', 'median'))
    
    df_var["range"]=df_var["max_value"] - df_var["min_value"]
    df_var = df_var.sort_values(by="range", ascending=ascending)
    
    get_data = df_var.iloc[count_start:count_start + no_records].reset_index()
    data_to_grab = pd.merge(labels_df, get_data, how="right", on=["county_name", "state_name"])
    
    return data_to_grab[["year", "state_ansi", "county_ansi", "county_name"]]

def check_blob_prefix_exists(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix, max_results=1)
    return any(blobs)

def batch_check_blobs(bucket_name, prefixes):
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_prefix = {executor.submit(check_blob_prefix_exists, bucket_name, prefix): prefix for prefix in prefixes}
        results = {}
        for future in as_completed(future_to_prefix):
            prefix = future_to_prefix[future]
            results[prefix] = future.result()
    return results

def img_name_composer(county, state_fips, year, month):
    image_name = f"{IMG_SOURCE_PREFIX}/{SCALE}/{county.capitalize()}_{state_fips}/{year}/{month}-{month+1}"
    return image_name

def blob_name_composer(county, state_fips, year, month):
    blob_name = f"{HIST_DEST_PREFIX}/{SCALE}/{county.capitalize()}_{state_fips}/{year}"
    return blob_name
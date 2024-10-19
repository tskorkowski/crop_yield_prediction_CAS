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
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import requests
from typing import Dict
from serving.constants import SCALE # meters per pixel


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


def get_input_image_ee(county: str, crop: int, year: int, month: int) -> ee.Image:
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
    county_geom = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("NAME", county))
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
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(mask_sentinel2_clouds)
            .select("B.*")
            .median()
            .unmask(1000)       
            .updateMask(cdl_image.eq(1))
            .float() 
        )
    s2_img = s2_img_unbounded.clip(county_geom)
    img_name = f"{county}_{year}_{month}-{month+1}"
    
    return {
            "image": s2_img,
            "image_name": img_name
    }
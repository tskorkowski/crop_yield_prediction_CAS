"""
This script is used to download Sentinel-2 imagery and create histograms for each image.
The script is designed to work with Google Earth Engine (GEE) and Google Cloud Storage (GCS).
"""

import json

import ee
import numpy as np
from google.cloud import storage

# Initialize GEE
ee.Initialize()

# Set up GCS client
storage_client = storage.Client()


def get_cropland_data(
    county, crop_type, start_year, end_year, harvest_start, harvest_end
):
    # Load datasets
    counties = ee.FeatureCollection("TIGER/2018/Counties")
    cdl = ee.ImageCollection("USDA/NASS/CDL")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    # Filter county
    county_geom = counties.filter(ee.Filter.eq("NAME", county)).first().geometry()

    # Filter years and harvest period
    def filter_year_harvest(year):
        start_date = ee.Date.fromYMD(
            year, harvest_start.get("month"), harvest_start.get("day")
        )
        end_date = ee.Date.fromYMD(
            year, harvest_end.get("month"), harvest_end.get("day")
        )
        return s2.filterDate(start_date, end_date)

    years = ee.List.sequence(start_year, end_year)
    s2_filtered = ee.ImageCollection(years.map(filter_year_harvest)).flatten()

    # Filter clouds and clip to county
    s2_filtered = s2_filtered.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).map(
        lambda img: img.clip(county_geom)
    )

    # Filter crop type and clip to crop area
    crop_mask = cdl.select("cropland").eq(crop_type)
    s2_clipped = s2_filtered.map(lambda img: img.updateMask(crop_mask))

    # Group images by month and reduce by mean
    def group_by_month(img):
        return ee.Feature(None, {"month": img.date().get("month"), "image": img})

    s2_monthly = (
        s2_clipped.map(group_by_month)
        .distinct("month")
        .map(lambda f: ee.Image(f.get("image")))
    )

    # Create histograms for each image
    def create_histograms(img):
        bands = img.bandNames()
        histograms = bands.map(
            lambda b: img.select([b])
            .reduceRegion(
                reducer=ee.Reducer.histogram(33, 1, 4999),
                geometry=county_geom,
                scale=30,
            )
            .get(b)
        )
        return ee.Feature(None, {"image": img, "histograms": histograms})

    s2_histograms = s2_monthly.map(create_histograms)

    # Save images and histograms to GCS
    def save_to_gcs(feat):
        img = ee.Image(feat.get("image"))
        histograms = ee.List(feat.get("histograms"))

        # Save image
        img_name = f"{county}_{crop_type}_{img.date().format('YYYY-MM').getInfo()}.tif"
        img_task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=img_name,
            bucket="your-bucket-name",
            fileNamePrefix=img_name,
            scale=30,
            region=county_geom,
        )
        img_task.start()

        # Save histograms
        hist_name = f"{county}_{crop_type}_{img.date().format('YYYY-MM').getInfo()}_histograms.json"
        hist_data = histograms.map(lambda h: ee.Dictionary(h).getInfo()).getInfo()
        bucket = storage_client.get_bucket("your-bucket-name")
        blob = bucket.blob(hist_name)
        blob.upload_from_string(json.dumps(hist_data))

    s2_histograms.map(save_to_gcs)


if __name__ == "__main__":
    # Example usage
    get_cropland_data(
        county="Fresno County",
        crop_type=1,  # Corn
        start_year=2020,
        end_year=2022,
        harvest_start={"month": 9, "day": 1},
        harvest_end={"month": 11, "day": 30},
    )

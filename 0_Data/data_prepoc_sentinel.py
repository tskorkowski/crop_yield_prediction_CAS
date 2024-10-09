"""
This script is used to download Sentinel-2 imagery and create histograms for each image.
The script is designed to work with Google Earth Engine (GEE) and Google Cloud Storage (GCS).
"""

import json
import logging
import multiprocessing as mp
import os
from datetime import datetime

import ee
import numpy as np
from google.cloud import storage
from google.cloud.exceptions import NotFound
from retry import retry

GCS_BUCKET = "vgnn"
PROJECT_ID = "supple-nature-370421"
CLOUD_COVER_THRESHOLD = 25

# Initialize GEE
ee.Initialize(
    project=PROJECT_ID, opt_url="https://earthengine-highvolume.googleapis.com"
)

# Set up GCS client
storage_client = storage.Client(project=PROJECT_ID)


# TODO: Iterating through counties
# TODO: Logging
# TODO: multiprocessing
# TODO: Error handling
# TODO: Documentation


def file_exists_in_gcs(blob_name, folder, bucket_name=GCS_BUCKET):
    """Check if a file exists in the specified GCS bucket and folder."""

    bucket = storage_client.bucket(bucket_name)
    full_path = f"{folder}/{blob_name}" if folder else blob_name
    blob = bucket.blob(full_path)
    try:
        blob.reload()
        return True
    except NotFound:
        return False


def mask_s2_clouds(image):
    # Select the QA60 band
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # Return the masked and scaled image
    return image.updateMask(mask)


def get_county_cropland_mask(county, crop_type, start_year, end_year):

    # Filter county
    county_geom = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("NAME", county))
        .geometry()
    )

    # Cropland data
    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterBounds(county_geom)
        .filter(ee.Filter.calendarRange(start_year, end_year, "year"))
        .select(0)
    )

    cdl_county_masked = cdl.map(
        lambda img: img.updateMask(img.remap([crop_type], [1], 0))
    )

    return cdl_county_masked, county_geom


def aggregate_s2_yearly_monthly(
    s2_collection, cropland_mask, start_year, end_year, seanson_start, seanson_end
):
    def process_year(year):
        # Filter the collection for the specific year

        year_crop_mask = cropland_mask.filter(
            ee.Filter.calendarRange(year, year, "year")
        ).first()

        # mask anything outside specific crop type
        year_collection = s2_collection.filter(
            ee.Filter.calendarRange(year, year, "year")
        ).map(lambda img: img.updateMask(year_crop_mask))

        # Function to aggregate images monthly
        def aggregate_monthly(month):
            return (
                year_collection.filter(ee.Filter.calendarRange(month, month, "month"))
                .limit(30)
                .median()
                .set("year", year)
                .set("month", month)
                .set("system:time_start", ee.Date.fromYMD(year, month, 1).millis())
            )

        # Create a list of months months in a season and map over them
        months = ee.List.sequence(seanson_start["month"], seanson_end["month"])
        months_aggregated = months.map(aggregate_monthly)
        return months_aggregated

    # Create a list of years and map over them
    years = ee.List.sequence(start_year, end_year)
    s2_monthly = years.map(process_year).flatten()
    return s2_monthly


def get_county_cropped_sentinel_data(
    county_geom, cropland_mask, start_year, end_year, seanson_start, seanson_end
):

    start_date = ee.Date.fromYMD(
        start_year, seanson_start["month"], seanson_start["day"]
    )
    end_date = ee.Date.fromYMD(end_year, seanson_end["month"], seanson_end["day"])

    # Get all band names
    all_bands = ee.Image(
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").first()
    ).bandNames()

    # Filter bands
    bands_of_interest = all_bands.filter(ee.Filter.stringStartsWith("item", "B"))

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(ee.Filter.date(start_date, end_date))
        .filterBounds(county_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_THRESHOLD))
        .map(mask_s2_clouds)
        .select(bands_of_interest)
    )

    s2_filtered = aggregate_s2_yearly_monthly(
        s2, cropland_mask, start_year, end_year, seanson_start, seanson_end
    )

    return s2_filtered


def get_cropland_data(
    county,
    crop_type,
    start_year,
    end_year,
    seanson_start,
    seanson_end,
    image_export=False,
    histogram_export=True,
):

    county_crop_mask, county_geom = get_county_cropland_mask(
        county, crop_type, start_year, end_year
    )
    sentinel_data = get_county_cropped_sentinel_data(
        county_geom, county_crop_mask, start_year, end_year, seanson_start, seanson_end
    ).flatten()

    # Create histograms for each image
    def create_histograms(obj):
        img = ee.Image(obj)
        histogram = img.reduceRegion(
            reducer=ee.Reducer.histogram(33, 50, 4999),
            geometry=county_geom,
            scale=30,
            maxPixels=10e9,
        )
        return ee.Feature(None, {"image": img, "histogram": histogram})

    # s2_histograms = sentinel_data.map(create_histograms)

    # Save images and histograms to GCS

    n_of_exports = sentinel_data.size().getInfo()
    for idx in range(n_of_exports):

        img = ee.Image(sentinel_data.get(idx))

        date = img.date().format("YYYY-MM-dd").getInfo()
        img_name = "_".join([county, str(crop_type), date])
        hist_name = img_name + "_hist"

        if histogram_export and not file_exists_in_gcs(hist_name, "histograms"):
            s2_histogram = create_histograms(img)
            hist = ee.Dictionary(s2_histogram.get("histogram"))
            hist_data = hist.getInfo()
            hist_json = json.dumps(hist_data, indent=2)

            blob_name = blob_name = f"histograms/{hist_name}"
            bucket = storage_client.get_bucket(GCS_BUCKET)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(hist_json, content_type="application/json")

        # Save image
        if image_export and not file_exists_in_gcs(img_name, "images"):
            try:
                img_task = ee.batch.Export.image.toCloudStorage(
                    image=img,
                    description=img_name,
                    bucket=GCS_BUCKET,
                    fileNamePrefix=f"images/{img_name}",
                    scale=30,
                    region=county_geom,
                )
                img_task.start()
            except ee.EEException as e:
                print(f"Export task failed: {e}")


if __name__ == "__main__":
    # Example usage
    get_cropland_data(
        county="Fresno",
        crop_type=1,  # Corn
        start_year=2017,
        end_year=2023,
        seanson_start={"month": 4, "day": 1},
        seanson_end={"month": 10, "day": 30},
        image_export=False,
        histogram_export=True,
    )

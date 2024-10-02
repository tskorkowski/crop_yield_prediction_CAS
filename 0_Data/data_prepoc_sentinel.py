"""
This script is used to download Sentinel-2 imagery and create histograms for each image.
The script is designed to work with Google Earth Engine (GEE) and Google Cloud Storage (GCS).
"""

import json
import os

import ee
import numpy as np
from google.cloud import storage

GCS_BUCKET = "vgnn"
PROJECT_ID = "supple-nature-370421"


# Initialize GEE
ee.Initialize(project=PROJECT_ID)

# Set up GCS client
storage_client = storage.Client(project=PROJECT_ID)


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


def get_county_cropland_mask(county, crop_type):

    # Load counties information
    counties = ee.FeatureCollection("TIGER/2018/Counties")

    # Filter county
    county_geom = counties.filter(ee.Filter.eq("NAME", county)).first().geometry()

    # Cropland data
    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterBounds(county_geom)
        .map(lambda img: img.clip(county_geom))
    )

    cdl_county_masked = cdl.map(
        lambda img: img.updateMask(img.remap([crop_type], [1], 0))
    )

    return cdl_county_masked, county_geom


def aggregate_s2_yearly_monthly(
    s2_collection, start_year, end_year, seanson_start, seanson_end
):
    def process_year(year):
        # Filter the collection for the specific year
        year_collection = s2_collection.filter(
            ee.Filter.calendarRange(year, year, "year")
        )

        # Function to aggregate images monthly
        def aggregate_monthly(month):
            return (
                year_collection.filter(ee.Filter.calendarRange(month, month, "month"))
                .limit(30)
                .map(mask_s2_clouds)
                .mean()
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
    s2_monthly = ee.ImageCollection(years.map(process_year).flatten())
    return s2_monthly


def get_county_sentinel_data(
    county_geom, start_year, end_year, seanson_start, seanson_end
):

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(county_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    s2_filtered = aggregate_s2_yearly_monthly(
        s2, start_year, end_year, seanson_start, seanson_end
    )

    return s2_filtered


def get_cropland_data(
    county, crop_type, start_year, end_year, seanson_start, seanson_end
):

    county_crop_mask, county_geom = get_county_cropland_mask(county, crop_type)
    sentinel_data = get_county_sentinel_data(
        county_geom, start_year, end_year, seanson_start, seanson_end
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
        return ee.Feature(county_geom, {"image": img, "histograms": histograms})

    s2_histograms = sentinel_data.map(create_histograms)

    # Save images and histograms to GCS
    def save_to_gcs(feat):
        img = ee.Image(feat.get("image"))
        histograms = ee.List(feat.get("histograms"))

        # Save image
        img_name = (
            ee.String(county)
            .cat("_")
            .cat(crop_type)
            .cat("_")
            .cat(img.date().format("YYYY-MM"))
            .cat(".tif")
        )
        img_task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=img_name,
            bucket=GCS_BUCKET,
            fileNamePrefix=img_name,
            scale=30,
            region=county_geom,
        )
        img_task.start()

        # Save histograms
        hist_name = (
            ee.String(county)
            .cat("_")
            .cat(crop_type)
            .cat("_")
            .cat(img.date().format("YYYY-MM"))
            .cat("_")
            .cat("histograms.json")
        )
        hist_data = histograms.map(lambda h: ee.Dictionary(h).getInfo()).getInfo()
        bucket = storage_client.get_bucket(GCS_BUCKET)
        blob = bucket.blob(hist_name)
        blob.upload_from_string(json.dumps(hist_data))

    s2_histograms.map(save_to_gcs)


if __name__ == "__main__":
    # Example usage
    get_cropland_data(
        county="Fresno",
        crop_type=1,  # Corn
        start_year=2020,
        end_year=2020,
        seanson_start={"month": 5, "day": 1},
        seanson_end={"month": 9, "day": 30},
    )

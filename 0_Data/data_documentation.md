# Data Documentation

# Google Earth Engine Data Extraction and Storage - data_prepoc_sentinel.py

## Overview
This Python script extracts agricultural data from Google Earth Engine (GEE) and stores it in Google Cloud Storage (GCS). It focuses on retrieving Sentinel-2 satellite imagery for specific crops in chosen counties, overlaying it with cropland data, and saving both images and histograms.

## Dependencies
- `ee`: Google Earth Engine Python API
- `numpy`: Numerical computing library
- `google.cloud.storage`: Google Cloud Storage client library

## Main Function: `get_cropland_data`

### Parameters
- `county` (str): Name of the county to analyze
- `crop_type` (int): Crop type code (e.g., 1 for corn)
- `start_year` (int): First year of the analysis period
- `end_year` (int): Last year of the analysis period
- `harvest_start` (dict): Start of harvest period (`{'month': int, 'day': int}`)
- `harvest_end` (dict): End of harvest period (`{'month': int, 'day': int}`)

### Workflow

1. **Data Loading**
   - Loads US Census Counties, USDA Cropland Data Layer, and Sentinel-2 data

2. **County Filtering**
   - Filters the specified county from the Counties dataset

3. **Time Period Filtering**
   - Filters Sentinel-2 data for the specified years and harvest periods

4. **Cloud Cover Filtering**
   - Removes images with >20% cloud cover

5. **Crop Area Clipping**
   - Clips images to the area of the specified crop type

6. **Monthly Aggregation**
   - Groups images by month and calculates the mean

7. **Histogram Creation**
   - Generates histograms for each band of the monthly images

8. **Data Storage**
   - Saves images as GeoTIFF files and histograms as JSON files in GCS

## Helper Functions

### `filter_year_harvest`
Filters Sentinel-2 data for a specific year and harvest period.

### `group_by_month`
Groups Sentinel-2 images by month.

### `create_histograms`
Creates histograms for each band of an image.

### `save_to_gcs`
Saves an image and its histograms to Google Cloud Storage.

## Usage Example
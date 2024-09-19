# To execute: python pull_modis.py -k keyfile.key

import ee
import time
import sys
from unidecode import unidecode
import argparse
import os.path
import geopandas as gpd
from datetime import datetime, timedelta


BUCKET_VM_REL = os.path.expanduser('~/bucket2/')

IMG_COLLECTIONS_1day = ['MODIS/061/MOD09GA', 'MODIS/061/MYD11A1', 'MODIS/061/MCD12Q1']
IMG_COLLECTIONS_8day = ['MODIS/061/MOD09A1', 'MODIS/061/MYD11A2', 'MODIS/061/MCD12Q1']
IMG_COLLECTION_BANDS = [[0, 1, 2, 3, 4, 5, 6, 12], [0, 4], [0]]
IMG_COLLECTION_CODES = ['sat', 'temp', 'cover']


#IMPORTANT: USA MODIS code optimized for pulling from subset of US states
#represented by the FIPS codes and postcodes below
USA_SOY_FIPS_CODES = {
    "29": "MO", "20": "KS", "31": "NE", "19": "IA", "38": "ND", "46": "SD",
    "27": "MN", "05": "AR", "17": "IL", "18": "IN", "39": "OH"
}

# Some lambda functions useful for the config options below
# For each lambda, r is a dictionary corresponding to a single feature, and l is a key of interest in that dictionary
# For example, Argentina features have keys "partido" and "provincia" for county and province, respectively.
remove_chars = "'()/&-"
translation_table = str.maketrans('', '', remove_chars)
CLEAN_NAME = lambda r, l: unidecode(r.get('properties').get(l)).lower().translate(translation_table).strip()
GET_FIPS = lambda r, l: USA_SOY_FIPS_CODES[r.get('properties').get(l)].lower()

## CONFIG OPTIONS
# To teach this script about a new country, add a new element of each of these five lists.

# "Regions" list: An easy identifier for the country, for use when invoking the script from the command line.
REGIONS = [
    'argentina',
    'brazil',
    'india',
    'usa',
    'ethiopia',
]

# "Boundary Filters": A rough bounding box for the entire country, to help GEE search for imagery faster
BOUNDARY_FILTERS = [
    [-74, -52, -54, -21],
    [-34, -34, -74, 6],
    [68, 6, 97.5, 37.2],
    [-80, 32, -104.5, 49],
    [33, 3.5, 48, 15.4],
]

# "Feature Collections": The path in Google Earth Engine to a shapefile table specifying a set of subdivisions of the country
FTR_COLLECTIONS = [
    'users/nikhilarundesai/cultivos_maiz_sembrada_1314',
    'users/nikhilarundesai/BRMEE250GC_SIR',
    'users/nikhilarundesai/India_Districts',
    'users/nikhilarundesai/US_Counties',
    'users/nikhilarundesai/ET_Admin2',
]

# "Feature Key Functions": Lambda functions that extract a human-readable name from the metadata for a single feature in the shapefile
FTR_KEY_FNS = [
    lambda region: CLEAN_NAME(region, 'partido') + "-" + CLEAN_NAME(region, 'provincia'), # ARG: "<county name>-<province name>"
    lambda region: CLEAN_NAME(region, 'NM_MESO') + "-brasil", # BR: "<mesoregion name>-brasil"
    lambda region: CLEAN_NAME(region, 'DISTRICT') + "-" + CLEAN_NAME(region, 'ST_NM'), # IN: "<district name>-<state name>"
    lambda region: CLEAN_NAME(region, 'NAME') + "-" + GET_FIPS(region, 'STATEFP'), # US: "<county name>-<state name>"
    lambda region: CLEAN_NAME(region, 'ADMIN2') + "-" +  CLEAN_NAME(region,'ADMIN1'), # ET: "<zone name>-<state name>"
]

# "Feature Filter Functions": Lambda function that uses metadata to determine whether imagery is worth pulling for a particular region.
# Only useful for US where we might otherwise pull imagery for thousands of irrelevant counties.
FTR_FILTER_FNS = [
    lambda region: True,
    lambda region: True,
    lambda region: True,
    lambda region: region.get('properties').get('STATEFP') in USA_SOY_FIPS_CODES, # US: only pull imagery from states with soy production
    lambda region: True,
]



USAGE_MESSAGE = 'Usage: python pull_modis.py <' + ', '.join(IMG_COLLECTION_CODES) + '> <' + \
  ', '.join(REGIONS) + '>' # + '<folder in bucket/ (optional)>'
NUM_ARGS = 3
NUM_OPTIONAL_ARGS = 1

# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer
def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    if composite_period ==1:
        current = current.select(IMG_COLLECTION_BANDS[img_collection_index]).toUint16()
    else:
        current = current.select(IMG_COLLECTION_BANDS[img_collection_index])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

def shapefile_to_ee_feature_collection(shapefile_path, province_name,region_column_name):
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[gdf[region_column_name].apply(lambda x: x.replace(' ', '')) == province_name]

    # Convert GeoDataFrame to ee.FeatureCollection
    features = []
    for _, row in gdf.iterrows():
        geom_type = row['geometry'].geom_type
        if geom_type == 'Polygon':
            # If it's a Polygon, use the exterior coordinates directly
            geom = ee.Geometry.Polygon([list(x) for x in row['geometry'].exterior.coords[:]])
        elif geom_type == 'MultiPolygon':
            # If it's a MultiPolygon, use the `geoms` property
            polygons = [list(poly.exterior.coords[:]) for poly in row['geometry'].geoms]
            geom = ee.Geometry.MultiPolygon(polygons)
        else:
            # Skip or handle other geometry types as needed
            print(f"Skipping unsupported geometry type: {geom_type}")
            continue

        feature = ee.Feature(geom)
        features.append(feature)
    return ee.FeatureCollection(features)


def export_to_drive_GEE(img, fname, folder, expregion, scale, eeuser=None): # This exports to Google Drive. 
    # print("export to drive")
    expcoord = expregion.geometry().coordinates().getInfo()[0]

    expconfig = dict(description=fname, folder=folder, fileNamePrefix=fname, dimensions=None, region=expcoord,
                    scale=float(scale), crs='EPSG:4326', crsTransform=None, maxPixels=1e13)
    task = ee.batch.Export.image.toDrive(image=img.clip(expregion), **expconfig)
    task.start()
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        time.sleep(10)
    print('Done.', task.status())

def export_to_gcs_bucket(img, fname, bucket, expregion, scale):
    expcoord = expregion.geometry().coordinates().getInfo()[0]

    expconfig = dict(
        description=fname,
        bucket=bucket,
        fileNamePrefix=fname,
        dimensions=None,
        region=expcoord,
        scale=float(scale),
        crs='EPSG:4326',
        crsTransform=None,
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task = ee.batch.Export.image.toCloudStorage(image=img.clip(expregion), **expconfig)
    task.start()
    while task.status()['state'] == 'RUNNING':
        print('Running...')
        time.sleep(10)
    print('Done.', task.status())

def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):  # Ignore comments and empty lines
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()  # This also removes comments from the value, if any
    return params

if __name__ == "__main__":

  ee.Authenticate()
   
  parser = argparse.ArgumentParser(description="Pull MODIS data for specified countries and imagery types.")
  parser.add_argument("-k", "--key_file", type=str, help="Path to the key file with input parameters.")
  args = parser.parse_args()

  if args.key_file:
        params = read_key_file(args.key_file)
        start_date = params.get('START_DATE')
        end_date = params.get('END_DATE')
        region_name = params.get('REGION')
        composite_period = int(params.get('COMPOSITE_PERIOD'))
        use_gee_shapefile = params.get('USE_GEE_SHAPEFILE') == '1'
        shapefile_path = params.get('SHAPEFILE_PATH')
        local_folder = params.get('DOWNLOAD_FOLDER')
        scale = params.get('SCALE')
        specified_regions = params.get('REGIONS', '').split(' ') 
        region_column_name = params.get('REGION_COLUMN_NAME')
        boundary_filters_shp = params.get('BOUNDARY_FILTER_SHP', '').split(' ') 
        drive_folder = params.get('DRIVE_FOLDER')
        project = params.get('PROJECT')

  else:
        print("Key file is required.")
        sys.exit(1)

  ee.Initialize(project=project)

  for collection_name in IMG_COLLECTION_CODES:
    img_collection_index = IMG_COLLECTION_CODES.index(collection_name)
    if composite_period == 1:
        image_collection = IMG_COLLECTIONS_1day[img_collection_index]
    elif composite_period == 8:
        image_collection = IMG_COLLECTIONS_8day[img_collection_index]
    else:
        print(":: ERROR: Wrong COMPOSITE_PERIOD. Only 1 and 8 are allowed. ")


    if use_gee_shapefile:
        rgn_idx = REGIONS.index(region_name)
        ftr_collection = FTR_COLLECTIONS[rgn_idx]
        boundary_filter = BOUNDARY_FILTERS[rgn_idx]
        ftr_key_fn = FTR_KEY_FNS[rgn_idx]
        ftr_filter_fn = FTR_FILTER_FNS[rgn_idx]

        county_region = ee.FeatureCollection(ftr_collection)
        feature_list = county_region.toList(1e5)
        feature_list_computed = feature_list.getInfo()
        all_regions = [feature['properties']['partido'] for feature in feature_list_computed if 'properties' in feature and 'partido' in feature['properties']]
        if specified_regions[0] != 'all':
            feature_list_computed = [ feature for feature in feature_list_computed if feature['properties']['partido'].lower() in specified_regions]
    else:
        if len(specified_regions)>1:
            print(":: Error: Only one region allowed when using a local shapefile.")
            print(":: Download stopped.")
            exit(1)
        feature_list_computed = shapefile_to_ee_feature_collection(shapefile_path,specified_regions[0],region_column_name)
        if len(feature_list_computed.getInfo()['features'])>1:
            print(":: Error: Something is wrong reading the local shapefile.")
            print(":: Download stopped.")
            exit(1)
        feature_list_computed = ee.Feature(feature_list_computed.getInfo()['features'][0])
        boundary_filter = [float(item) for item in boundary_filters_shp]

    imgcoll = ee.ImageCollection(image_collection) \
        .filterBounds(ee.Geometry.Rectangle(boundary_filter))\
        .filterDate(start_date,end_date)
    img=imgcoll.iterate(appendBand)
    img=ee.Image(img)

    if img_collection_index != 1: #temperature index <<< is this min max filtering needed?
        img_0=ee.Image(ee.Number(0))
        img_5000=ee.Image(ee.Number(5000))

        img=img.min(img_5000)
        img=img.max(img_0)

    keys_with_issues = []
    count_already_downloaded = 0
    count_filtered = 0

    if use_gee_shapefile:
        for idx, region in enumerate(feature_list_computed):
            if not ftr_filter_fn(region):
                count_filtered += 1
                continue

            subunit_key = ftr_key_fn(region)
            file_name = region_name + '_' + collection_name + '_' + subunit_key + "_" + start_date + "_" + end_date
            if drive_folder is not None and \
            os.path.isfile(os.path.join(BUCKET_VM_REL, drive_folder, file_name + '.tif')):
                print(subunit_key, 'already downloaded. Continuing...')
                count_already_downloaded += 1
                continue
            try:
                export_to_drive_GEE(img, file_name, drive_folder, ee.Feature(region), scale)
            except KeyboardInterrupt:
                print('received SIGINT, terminating execution')
                break
            except Exception as e:
                print('issue ({})'.format(subunit_key, str(e)))
                keys_with_issues.append((subunit_key, str(e)))
    else:
        file_name = region_name + '_' + collection_name + '_' + start_date + "_" + end_date
        if drive_folder is not None and \
        os.path.isfile(os.path.join(BUCKET_VM_REL, drive_folder, file_name + '.tif')):
                print(subunit_key, 'already downloaded. Continuing...')
                count_already_downloaded += 1
                continue
        try:
            export_to_drive_GEE(img, file_name, drive_folder, feature_list_computed, scale)
        except KeyboardInterrupt:
            print('received SIGINT, terminating execution')
            break
        except Exception as e:
            print('issue ({})'.format(str(e)))
            keys_with_issues.append((str(e)))            

    if use_gee_shapefile:
        print('Successfully ordered', len(feature_list_computed)-len(keys_with_issues)-count_already_downloaded-count_filtered, 'new tifs from GEE')
    else:
        print('Successfully ordered', 1-len(keys_with_issues)-count_already_downloaded-count_filtered, 'new tifs.')
    print('Already had', count_already_downloaded)
    print('Failed to order', len(keys_with_issues))
    print('Filtered', count_filtered)

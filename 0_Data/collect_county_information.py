import ee

GCS_BUCKET = "vgnn"
PROJECT_ID = "supple-nature-370421"
CLOUD_COVER_THRESHOLD = 25

# Initialize GEE
ee.Initialize(
    project=PROJECT_ID, opt_url="https://earthengine-highvolume.googleapis.com"
)


def export_county_geometries():

    counties = ee.FeatureCollection("TIGER/2018/Counties")

    def get_county_geometries(feat):
        return ee.Feature(None, {"name": feat.get("NAME"), "geometry": feat.geometry()})

    county_geoms = ee.FeatureCollection(counties.map(get_county_geometries))

    task = ee.batch.Export.table.toCloudStorage(
        collection=county_geoms,
        description="USA County Geometries",
        bucket=GCS_BUCKET,
        fileNamePrefix=r"geometries/USA_Counties",
        fileFormat="GeoJSON",
    )

    task.start()


def export_couty_cropland_masks(start_year, end_year, crop_type, county):

    # Filter county
    county_geom = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("NAME", county))
        .geometry()
    )

    # Cropland data - image collection with specific crops masked
    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterBounds(county_geom)
        .select("cropland")
        .filter(ee.Filter.calendarRange(start_year, end_year, "year"))
        .map(lambda img: img.updateMask(img.remap([crop_type], [1], 0)))
    )

    cdl_county_masked = cdl.map(
        lambda img: img.updateMask(img.remap([crop_type], [1], 0))
    )

    return cdl_county_masked, county_geom

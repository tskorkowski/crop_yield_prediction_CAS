"""
This script is used to get counties interesting from the perspective of production of selected commodity.
In this case, we are interested in corn.
"""

import io
import json
import os
from typing import List

import pandas as pd
from nasspython.nass_api import nass_data


def get_nass_data(api_key: str, commodity: str, years: List[int]) -> pd.DataFrame:
    nass_data(api_key)

    data_frames = []

    for year in years:
        params = {
            "source_desc": "SURVEY",
            "api_key": api_key,
            "commodity_desc": commodity,
            "year": year,
            "freq_desc": "ANNUAL",
            "agg_level_desc": "COUNTY",
            "format": "CSV",
            "statisticcat_desc": "YIELD",
            "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        }

        data = pd.read_csv(io.StringIO(nass_data(**params).strip()))
        data_frames.append(data)
    return pd.concat(data_frames)


if __name__ == "__main__":
    api_key = os.getenv("NASS_API_KEY")
    COMMODITY = "CORN"
    years = list(range(2018, 2019))  # Adjust the range as needed

    corn_data = get_nass_data(api_key, COMMODITY, years)

    # Step 2: Calculate the average crop yield per county for each year
    corn_data["YIELD"] = pd.to_numeric(corn_data["Value"])

    # TODO: does it make sense to calculate average per year, I would assume production is
    # yearly to begin with
    avg_yield_per_county = (
        corn_data.groupby(["County", "State", "Year"])["YIELD"].mean().reset_index()
    )

    # Step 3: Identify counties with large corn fields based on a threshold
    corn_acreage = corn_data[corn_data["statisticcat_desc"] == "AREA HARVESTED"]
    corn_acreage["AREA"] = pd.to_numeric(corn_acreage["Value"])
    large_corn_counties = (
        corn_acreage.groupby(["county_name", "state_name"])["AREA"]
        .mean()
        .nlargest(500)
        .reset_index()
    )

    # Step 4: Compare actual crop yield with expected yield and flag lower than expected cases
    merged_data = pd.merge(
        avg_yield_per_county, large_corn_counties, on=["county_name", "state_name"]
    )
    merged_data["EXPECTED_YIELD"] = merged_data.groupby("state_name")[
        "YIELD"
    ].transform("mean")
    merged_data["LOWER_THAN_EXPECTED"] = (
        merged_data["YIELD"] < 0.9 * merged_data["EXPECTED_YIELD"]
    )

    # Step 5: Prepare the final dataset
    final_data = merged_data[
        [
            "county_name",
            "state_name",
            "year",
            "YIELD",
            "AREA",
            "EXPECTED_YIELD",
            "LOWER_THAN_EXPECTED",
        ]
    ]
    print(final_data.info())
    print(final_data)

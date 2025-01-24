"""
@inproceedings{fudong:kdd24:crop_net,
  author       = {Fudong Lin and Kaleb Guillot and Summer Crawford and Yihe Zhang and Xu Yuan and Nian{-}Feng Tzeng},
  title        = {An Open and Large-Scale Dataset for Multi-Modal Climate Change-aware Crop Yield Predictions},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining (KDD)},
  pages        = {5375--5386},
  year         = {2024}
}

weather data comes from the abovementioned paper.
https://huggingface.co/datasets/CropNet/CropNet
"""

import json
import os
import re
from collections import defaultdict

import polars as pl

# Dictionary of states and territories with their abbreviations

state_to_fips = {
    "AL": 1,
    "AK": 2,
    "AZ": 4,
    "AR": 5,
    "CA": 6,
    "CO": 8,
    "CT": 9,
    "DE": 10,
    "DC": 11,
    "FL": 12,
    "GA": 13,
    "HI": 15,
    "ID": 16,
    "IL": 17,
    "IN": 18,
    "IA": 19,
    "KS": 20,
    "KY": 21,
    "LA": 22,
    "ME": 23,
    "MD": 24,
    "MA": 25,
    "MI": 26,
    "MN": 27,
    "MS": 28,
    "MO": 29,
    "MT": 30,
    "NE": 31,
    "NV": 32,
    "NH": 33,
    "NJ": 34,
    "NM": 35,
    "NY": 36,
    "NC": 37,
    "ND": 38,
    "OH": 39,
    "OK": 40,
    "OR": 41,
    "PA": 42,
    "RI": 44,
    "SC": 45,
    "SD": 46,
    "TN": 47,
    "TX": 48,
    "UT": 49,
    "VT": 50,
    "VA": 51,
    "WA": 53,
    "WV": 54,
    "WI": 55,
    "WY": 56,
}

fips_to_state = {fips: state for state, fips in state_to_fips.items()}

histogram_path = (
    r"C:\Users\tskor\Documents\data\histograms\60_buckets_9_bands_60_res\60"
)

test_path = r"C:\Users\tskor\Documents\data\histograms\60_buckets_9_bands_60_res\test"


def iterate_histogram_folders(base_path) -> dict:
    # Go through all collected histograms to pick only the weather data that is needed

    pattern = r"(\w+)_(\d+)"
    compiled_re = re.compile(pattern)
    weather_req = defaultdict(lambda: defaultdict(list))

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        # Check if the current path is a directory
        if os.path.isdir(folder_path):
            county_name, state_fips = compiled_re.match(folder_name).groups()
            # Iterate through the subfolders one level below'
            state_abbreviation = fips_to_state[int(state_fips)]
            for year in os.listdir(folder_path):
                if (os.path.isdir(os.path.join(folder_path, year))) & (
                    int(year) >= 2017
                ):
                    weather_req[year][state_abbreviation].append(county_name)

    return weather_req


def preproc_weather_signle_file(
    path: str, col_to_filter: str, col_filter: list
) -> pl.DataFrame:
    """
    Steps:
    1. Read the file
    2. Drop NA values - monthly data at the bottom of the file
    3. Filter counties
    """
    df = pl.read_csv(path, null_values=["N/A"]).drop_nans()

    # Convert both the column and the filter values to lowercase
    df = df.with_columns(pl.col(col_to_filter).str.to_lowercase())
    col_filter = [value.lower() for value in col_filter]

    return df.filter(pl.col(col_to_filter).is_in(col_filter))


def try_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value


def preproc_weather_aggregation(
    df_to_combine: [pl.DataFrame], key: str, key_id: str = "Month"
) -> pl.DataFrame:

    combined_weather_data = pl.concat(df_to_combine)
    aggregated_weather_data = combined_weather_data.group_by(
        ["County", "FIPS Code"]
    ).agg(
        [
            pl.col("Max Temperature (K)").max().alias("Temp_Max"),
            pl.col("Min Temperature (K)").min().alias("Temp_Min"),
            pl.col("Avg Temperature (K)").median().alias("Temp_Median"),
            pl.col("Precipitation (kg m**-2)").sum().alias("Precipitation_Total"),
            pl.col("Precipitation (kg m**-2)").median().alias("Precipitation_Median"),
            pl.col("Precipitation (kg m**-2)")
            .quantile(0.9)
            .alias("Precipitation_90th_Percentile"),
            pl.col("Precipitation (kg m**-2)")
            .quantile(0.1)
            .alias("Precipitation_10th_Percentile"),
            pl.col("Relative Humidity (%)").mean().alias("Relative_Humidity_Mean"),
            pl.col("Wind Speed (m s**-1)").max().alias("Wind_Speed_Max"),
            pl.col("Downward Shortwave Radiation Flux (W m**-2)")
            .max()
            .alias("Downward_Shortwave_Radiation_Flux_Max"),
            pl.col("Downward Shortwave Radiation Flux (W m**-2)")
            .min()
            .alias("Downward_Shortwave_Radiation_Flux_Min"),
            pl.col("Downward Shortwave Radiation Flux (W m**-2)")
            .median()
            .alias("Downward_Shortwave_Radiation_Flux_Median"),
            pl.col("Vapor Pressure Deficit (kPa)")
            .median()
            .alias("Vapor_Pressure_Deficit_Median"),
            pl.col("FIPS Code").first().alias("FIPS"),
        ]
    )
    key = try_convert_to_int(key)
    aggregated_weather_data = aggregated_weather_data.with_columns(
        pl.lit(key).alias(key_id)
    )

    return aggregated_weather_data


def save_groups_to_csv(
    df: pl.DataFrame,
    group_column: str,
    output_dir: str,
    group_filter: list = None,
    file_name_suffix: str = None,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert both the column and the filter values to lowercase
    df = df.with_columns(pl.col(group_column).str.to_lowercase())
    group_filter_lower = [value.lower() for value in group_filter]

    # Perform the case-insensitive is_in operation
    df = df.filter(pl.col(group_column).is_in(group_filter_lower))

    # Get unique groups
    unique_groups = df.get_column(group_column).unique()

    # Split and save each group
    for group in unique_groups:
        # Filter for this group
        group_df = df.filter(pl.col(group_column) == group)
        fips = group_df.get_column("FIPS").unique().to_list()[0]
        # Create filename (clean group name if needed)

        filename = f"weather_data-{group}"

        if file_name_suffix:
            filename += f"-{file_name_suffix}"

        filename += f"-{fips}.csv"
        filepath = os.path.join(output_dir, filename)

        # Save to CSV
        group_df.write_csv(filepath)
        print(f"Saved group {group} to {filepath}")


def select_weather_files(month: int, file_list: list) -> list:
    """
    Select weather files for a specific months
    """

    file_name_pattern = r".+-(\d+).csv"
    compiled_re = re.compile(file_name_pattern)

    return [file for file in file_list if int(file.split("-")[1]) == month]


def load_weather_information(json_file_path):
    """
    Load weather information from a JSON file into a dictionary.
    """
    with open(json_file_path, "r") as f:
        weather_information = json.load(f)
    return weather_information


if __name__ == "__main__":
    # Load weather information from JSON
    json_file_path = (
        r"C:\Users\tskor\Documents\GitHub\inovation_project\2_Data\weather_req.json"
    )
    OUTPUT_DIR = r"C:\Users\tskor\Documents\data\WRF-HRRR\split_by_county_and_year"
    weather_information = load_weather_information(json_file_path)

    weather_data_path = r"C:\Users\tskor\Documents\data\WRF-HRRR"

    # collect information about which counties in which states are of interest
    # weather_information = iterate_histogram_folders(histogram_path)
    # with open(
    #     r"C:\Users\tskor\Documents\GitHub\inovation_project\2_Data\weather_req.json",
    #     "w",
    # ) as f:
    #     json.dump(weather_information, f, indent=4, sort_keys=True)

    data_generator = (
        (year, state, counties)
        for year, values in weather_information.items()
        for state, counties in values.items()
    )

    # Compile a regex pattern to match files ending with 05, 06, 07, 08, 09, or 10
    file_pattern = re.compile(r".*-(05|06|07|08|09|10)\.csv$")

    for year, state, counties in data_generator:
        aggregated_monthly_data = pl.DataFrame()
        weather_files_dir = os.path.join(weather_data_path, year, state)

        # Filter files using the regex pattern
        files_to_process = [
            file_to_process
            for file_to_process in os.listdir(weather_files_dir)
            if file_pattern.match(file_to_process)
        ]

        files_to_process_sorted = sorted(files_to_process)

        # Process files in pairs
        for i in range(0, len(files_to_process_sorted), 2):
            # Get the current pair of files
            file_pair = files_to_process_sorted[i : i + 2]

            # Read the files into DataFrames
            weather_data_files = [
                pl.read_csv(
                    os.path.join(weather_files_dir, file), null_values=["N/A"]
                ).drop_nans()
                for file in file_pair
            ]

            key = file_pattern.match(file_pair[0]).groups()[0]
            # Apply the aggregation to the pair of DataFrames
            post_proc_weather_data = preproc_weather_aggregation(
                weather_data_files, key
            )

            aggregated_monthly_data = pl.concat(
                [aggregated_monthly_data, post_proc_weather_data]
            )

        save_groups_to_csv(
            aggregated_monthly_data,
            "County",
            OUTPUT_DIR,
            weather_information[year][state],
            f"{year}-{state}",
        )

# TO DO: create function to fetch polars data frames from the specific folders

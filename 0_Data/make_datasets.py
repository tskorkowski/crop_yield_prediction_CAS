# To execute: python3 make_datasets.py -k keyfile.key

from constants import NUM_IMGS_PER_YEAR, GBUCKET, LOCAL_DATA_DIR
import sys
import numpy as np
import random
import os
from os.path import isfile, join, basename, normpath
import pandas as pd
import sklearn
import argparse
from datetime import datetime

DEFAULT_TEST_POOL_FRACTION = 0.2
DEFAULT_DEV_FRAC_OF_TRAIN = 0.2  # NOT IMPLEMENTED 0.2

HARVEST = "Year"
CROP_FIELD = "Crop"
PROVINCE = "Region1"
DEPARTMENT = "Region2"
YIELD = "Yield"
HIST_SUFFIX = "histogram.npy"


def is_within_range_first_year(date, start_date, end_date, reference_year):
    date = datetime.strptime(date, "%Y-%m-%d")
    if date.year != reference_year:
        return False

    start_date = datetime(reference_year, start_date.month, start_date.day)
    end_date = datetime(reference_year, end_date.month, end_date.day)

    return start_date <= date <= end_date


def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith(
                "#"
            ):  # Ignore comments and empty lines
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[
                        0
                    ].strip()  # This also removes comments from the value, if any
    return params


def return_files(path):
    result = []
    sys.stdout.write("list {}: ".format(path))
    paths = os.listdir(path)
    sys.stdout.write("{} paths (each '.' represents a path)\n".format(len(paths)))
    for f in paths:
        sys.stdout.write(".")
        sys.stdout.flush()
        if isfile(join(path, f)) and not f.startswith("."):
            result.append(f)
    sys.stdout.write("\n")
    return result


def return_yield_file(country):
    return join(LOCAL_DATA_DIR, "{}_yields_standardized.csv".format(country.lower()))


def return_filtered_regions_file(country, crop):
    return join(
        LOCAL_DATA_DIR, country.lower() + "_" + crop.lower() + "_filtered_regions.txt"
    )


def get_yield(df, harvest):
    sub_df = df[HARVEST] == harvest
    result = df[sub_df]
    if result.shape[0] == 0:
        return None
    else:
        return result.iloc[0][YIELD]


def get_location(df, region_2, region_1):
    sub_df = (df[DEPARTMENT] == region_2) & (df[PROVINCE] == region_1)
    result = df[sub_df]
    if result.shape[0] == 0:
        return None
    else:
        return result.iloc[0]["Loc1"], result.iloc[0]["Loc2"]


def parse_filename(file_key, country):
    sat_prefix = country.lower() + "_sat_"
    if file_key.find(sat_prefix) != -1:
        file_key = file_key[file_key.find(sat_prefix) + len(sat_prefix) :]

    file_key = file_key[: file_key.find("_")]
    region_2 = file_key[: file_key.find("-")]
    region_1 = file_key[file_key.find("-") + 1 :]
    return region_2, region_1


def remove_months_outside_harvest(
    f_data,
    beginning_offset,
    season_len,
    harv_begin,
    harv_end,
    indexes_harvest,
    dataset_year_begin,
    dataset_year_end,
    ndvi=False,
):
    f_data_by_year = []
    # print('remove_months_outside_harvest')
    for idx in range(0, f_data.shape[1 if not ndvi else 0], NUM_IMGS_PER_YEAR):
        if not ndvi:
            f_data_by_year.append(f_data[:, idx : idx + NUM_IMGS_PER_YEAR, :])
        else:
            f_data_by_year.append(f_data[idx : idx + NUM_IMGS_PER_YEAR])
    if f_data_by_year[-1].shape[1 if not ndvi else 0] != NUM_IMGS_PER_YEAR:
        print(
            "Error in remove_months_outside_harvest. Take a closer look to remove_months_outside_harvest_old."
        )
        exit()

    if not ndvi:
        trimmed_to_season_end = [x[:, indexes_harvest, :] for x in f_data_by_year]
    else:
        trimmed_to_season_end = [x[indexes_harvest] for x in f_data_by_year]
    beginning_year = harv_begin - dataset_year_begin
    total_number_of_years = min(harv_end, dataset_year_end) - harv_begin + 1
    # print(np.shape(trimmed_to_season_end))
    return trimmed_to_season_end[
        beginning_year : beginning_year + total_number_of_years
    ]


def remove_months_outside_harvest_old(
    f_data,
    beginning_offset,
    season_len,
    harv_begin,
    harv_end,
    indexes_harvest,
    dataset_year_begin,
    dataset_year_end,
    ndvi=False,
):
    f_data_by_year = []
    # have to set offset at the beginning of the year to remove data when not in season
    if not ndvi:
        f_data = f_data[:, beginning_offset:, :]
    else:
        f_data = f_data[beginning_offset:]
    for idx in range(0, f_data.shape[1 if not ndvi else 0], NUM_IMGS_PER_YEAR):
        if not ndvi:
            f_data_by_year.append(f_data[:, idx : idx + NUM_IMGS_PER_YEAR, :])
        else:
            f_data_by_year.append(f_data[idx : idx + NUM_IMGS_PER_YEAR])
    if (
        f_data_by_year[-1].shape[1 if not ndvi else 0] != NUM_IMGS_PER_YEAR
    ):  # trim off excess
        f_data_by_year = f_data_by_year[:-1]
    # our histograms in total comprise the harvest in 2003 to harvest to 2016
    if not ndvi:
        trimmed_to_season_end = [x[:, :season_len, :] for x in f_data_by_year]
    else:
        trimmed_to_season_end = [x[:season_len] for x in f_data_by_year]
    beginning_year = harv_begin - dataset_year_begin
    total_number_of_years = min(harv_end, dataset_year_end) - harv_begin + 1
    return trimmed_to_season_end[
        beginning_year : beginning_year + total_number_of_years
    ]


def sort_harvest_year_strings(years):
    return sorted(years, key=lambda year_str: int(year_str))  # chronological order


def calculate_offset(crop, country):
    # assuming that all histograms start 2002-07-31
    # we're going to assume 3 images (~24 days) for every month we cut out, and 4 images (~32 days) for every month we leave in
    if crop == "soybeans" and country.lower() == "argentina":
        # Soy is October to June, but forced to be length of 32
        return 6, 32
    elif crop == "soybeans" and country.lower() == "brazil":
        # soybeans is September to April http://www.soybeansandcorn.com/Brazil-Crop-Cycles
        return 3, 32
    else:
        return 15, 32


def make_files_set_test(
    hist_dir,
    crop,
    country,
    harv_begin,
    harv_end,
    harvest_phase_begin,
    harvest_phase_end,
    season_frac,
    test_region_1s,
    test_years,
    test_pool_frac,
    filter_regions,
    filter_years,
    exclude,
    use_skip_file,
    verbose,
    train_fraction_keep,
    dev_frac,
    scale_factor,
    use_gee_shapefile,
    region_input,
    dataset_year_begin,
    dataset_year_end,
    year_to_skip,
):
    harvest_years = [
        (year) for year in range(harv_begin, harv_end + 1) if year != year_to_skip
    ]
    if verbose:
        print("these are our harvest years: ")
        print(harvest_years)
        print

    if test_years is not None:
        test_years = sort_harvest_year_strings(test_years)
    (
        train_hists,
        train_ndvi,
        train_yields,
        train_keys,
        years_for_train_data,
        train_locs,
    ) = ([], [], [], [], [], [])
    test_hists, test_ndvi, test_yields, test_keys, years_for_test_data, test_locs = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    yields_file = return_yield_file(country)
    if verbose:
        print("Using the yields file named: " + yields_file)
    yields = pd.read_csv(yields_file)
    crop_sub_df = yields[CROP_FIELD] == crop
    yields = yields[crop_sub_df]  # filter out crops

    location_file = join(os.getcwd(), LOCAL_DATA_DIR, country.lower() + "_loc.csv")
    filtered_regions_file = return_filtered_regions_file(country, crop)

    locations = None
    if isfile(location_file):
        locations = pd.read_csv(location_file)
        if verbose:
            print("Locations file found: " + location_file)
    elif verbose:
        print("Locations file not found.")

    low_production_regions_to_skip = None
    if isfile(filtered_regions_file):
        low_production_regions_to_skip = set(
            s.lower() for s in open(filtered_regions_file).read().split("\n")
        )
        if verbose:
            print("Using skipfile: " + filtered_regions_file)
            print(
                "Skipping these regions: " + ", ".join(low_production_regions_to_skip)
            )

    # Get all tifs in the data directory
    all_hist_files = [
        f for f in return_files(hist_dir) if f[-len(HIST_SUFFIX) :] == HIST_SUFFIX
    ]

    count = 0
    failed_regions = []
    for filename in all_hist_files:
        print("processing file {}".format(filename))

        region_2, region_1 = parse_filename(filename, country)
        region_sub_df = (yields[DEPARTMENT] == region_2) & (
            yields[PROVINCE] == region_1
        )  # & crop_sub_df
        yields_temp = yields[region_sub_df]  # filter out other provinces

        location_for_region = None
        if locations is not None:
            location_for_region = get_location(locations, region_2, region_1)
            if location_for_region is None:
                print("Location not found for region: " + region_2 + " " + region_1)
                continue

        if filter_regions is not None and region_1.lower() in filter_regions:
            continue

        region_key = region_2.lower() + "-" + region_1.lower()

        if (
            low_production_regions_to_skip is not None
            and use_skip_file
            and region_key in low_production_regions_to_skip
        ):
            continue

        # calculate the appropriate begininning and ending offset
        beginning_offset, season_len = calculate_offset(
            crop, country
        )  # deprecate season_len... use universal SEASON_LEN
        beginning_offset = 0
        hist_data = np.load(join(hist_dir, filename))
        dates = hist_data[0, :, -1]
        reference_year = datetime.strptime(dates[0], "%Y-%m-%d").year
        indexes_harvest = [
            i
            for i, date in enumerate(dates)
            if is_within_range_first_year(
                date,
                datetime.strptime(harvest_phase_begin, "%m-%d"),
                datetime.strptime(harvest_phase_end, "%m-%d"),
                reference_year,
            )
        ]
        hist_data = hist_data[:, :, :-1]  # Remove the dates.
        season_len = int(season_frac * season_len)
        # print(np.shape(hist_data))
        truncated_f_data_by_year = remove_months_outside_harvest(
            hist_data,
            beginning_offset,
            season_len,
            harv_begin,
            harv_end,
            indexes_harvest,
            dataset_year_begin,
            dataset_year_end,
        )
        if year_to_skip and year_to_skip >= harv_begin and year_to_skip <= harv_end:
            skip_index = year_to_skip - harv_begin
            if skip_index < len(truncated_f_data_by_year):
                del truncated_f_data_by_year[skip_index]
        assert len(truncated_f_data_by_year) == len(harvest_years)
        ndvi_path = join(hist_dir, "{}ndvi.npy".format(filename[: -len(HIST_SUFFIX)]))
        if isfile(ndvi_path):
            ndvi_data = np.load(ndvi_path)
        else:  # dummy data
            ndvi_data = np.full(hist_data.shape[1], None)
        truncated_ndvi_data_by_year = remove_months_outside_harvest(
            ndvi_data,
            beginning_offset,
            season_len,
            harv_begin,
            harv_end,
            indexes_harvest,
            dataset_year_begin,
            dataset_year_end,
            ndvi=True,
        )
        if year_to_skip and year_to_skip >= harv_begin and year_to_skip <= harv_end:
            skip_index = year_to_skip - harv_begin
            if skip_index < len(truncated_ndvi_data_by_year):
                del truncated_ndvi_data_by_year[skip_index]
        assert len(truncated_ndvi_data_by_year) == len(harvest_years)

        success_years = []

        for i, year in enumerate(harvest_years):
            if filter_years is not None and year == int(filter_years):
                continue
            nextval = get_yield(yields_temp, year)
            if nextval is not None:
                success_years.append(str(year))
                if (
                    (
                        test_years is None
                        and test_region_1s is not None
                        and region_1.lower() in test_region_1s
                    )
                    or (
                        test_region_1s is None
                        and test_years is not None
                        and str(year) in test_years
                    )
                    or (test_years is not None and test_region_1s is not None)
                    and (region_1.lower() in test_region_1s and str(year) in test_years)
                ):
                    test_hists.append(truncated_f_data_by_year[i])
                    test_ndvi.append(truncated_ndvi_data_by_year[i])
                    test_yields.append(nextval * scale_factor)
                    test_keys.append(region_2 + "_" + region_1 + "_" + str(year))
                    years_for_test_data.append(str(year))
                    test_locs.append(location_for_region)
                else:
                    if exclude:
                        if test_years is not None:
                            int_year = int(year)
                            int_first_test_year = int(test_years[0])
                            if (
                                int_year > int_first_test_year
                            ):  # don't train on years beyond first test year
                                continue
                    train_hists.append(truncated_f_data_by_year[i])
                    train_ndvi.append(truncated_ndvi_data_by_year[i])
                    train_yields.append(nextval * scale_factor)
                    train_keys.append(region_2 + "_" + region_1 + "_" + str(year))
                    years_for_train_data.append(str(year))
                    train_locs.append(location_for_region)
        if verbose:
            if len(success_years) == 0:
                print(region_2, region_1, "skipped - no yield data found")
                failed_regions.append(region_2 + "_" + region_1)
                continue
            print(region_2, region_1, "successful -", ", ".join(success_years))
            count += 1
            if count % 50 == 0:
                print("Processed", str(count), "histograms successfully")
    if verbose:
        print()
        print("Failed to process", ", ".join(failed_regions))
        print()
        print("Shuffling datasets...")
    random.seed(12)
    # shuffle train and test sets

    test = list(
        zip(
            test_hists,
            test_ndvi,
            test_yields,
            test_keys,
            years_for_test_data,
            test_locs,
        )
    )
    test = sklearn.utils.shuffle(test)
    train = list(
        zip(
            train_hists,
            train_ndvi,
            train_yields,
            train_keys,
            years_for_train_data,
            train_locs,
        )
    )

    train = sklearn.utils.shuffle(train)
    keep = int(train_fraction_keep * len(train))
    train = train[:keep]

    # sample from train if no test year or province
    if test_years is None and test_region_1s is None:
        if verbose:
            print(
                "No test year or test province specified. Sampling test set from training set..."
            )
        test_size = int(len(train) * test_pool_frac)
        if test_size > 0:
            test = train[:test_size]
            train = train[test_size:]
        else:
            test = []

    dev_size = int(len(train) * dev_frac)
    if dev_size > 0:
        dev = train[:dev_size]
        train = train[dev_size:]
    else:
        dev = []

    return train, dev, test


def run(
    hist_dir,
    target_dir,
    crop,
    country,
    harv_begin,
    harv_end,
    harvest_phase_begin,
    harvest_phase_end,
    season_frac,
    test_region_1s,
    test_years,
    test_pool_frac,
    filter_regions,
    filter_years,
    dev_frac,
    exclude,
    remake,
    use_skip_file,
    verbose,
    train_fraction_keep,
    scale_factor,
    use_gee_shapefile,
    region_input,
    dataset_year_begin,
    dataset_year_end,
    year_to_skip,
):
    directory = target_dir

    if "Prediction" in directory:
        test_pool_frac, dev_frac = 0.0, 0.0

    if not os.path.exists(directory):
        os.makedirs(directory)
    if verbose:
        print("Saving into directory: " + directory)
        print()

    train_hists_save_path = join(directory, "train_hists.npz")
    train_ndvi_save_path = join(directory, "train_ndvi.npz")
    train_yields_save_path = join(directory, "train_yields.npz")
    train_keys_save_path = join(directory, "train_keys.npz")
    train_years_save_path = join(directory, "train_years.npz")
    train_locs_save_path = join(directory, "train_locs.npz")
    print(train_locs_save_path)

    dev_hists_save_path = join(directory, "dev_hists.npz")
    dev_ndvi_save_path = join(directory, "dev_ndvi.npz")
    dev_yields_save_path = join(directory, "dev_yields.npz")
    dev_keys_save_path = join(directory, "dev_keys.npz")
    dev_years_save_path = join(directory, "dev_years.npz")
    dev_locs_save_path = join(directory, "dev_locs.npz")

    test_hists_save_path = join(directory, "test_hists.npz")
    test_ndvi_save_path = join(directory, "test_ndvi.npz")
    test_yields_save_path = join(directory, "test_yields.npz")
    test_keys_save_path = join(directory, "test_keys.npz")
    test_years_save_path = join(directory, "test_years.npz")
    test_locs_save_path = join(directory, "test_locs.npz")

    pred_hists_save_path = join(directory, "hists.npz")
    pred_ndvi_save_path = join(directory, "ndvi.npz")
    pred_yields_save_path = join(directory, "yields.npz")
    pred_keys_save_path = join(directory, "keys.npz")
    pred_years_save_path = join(directory, "years.npz")
    pred_locs_save_path = join(directory, "locs.npz")

    if remake or not isfile(
        test_hists_save_path
    ):  # assume if test_hists is available other files are as well
        train, dev, test = make_files_set_test(
            hist_dir,
            crop,
            country,
            harv_begin,
            harv_end,
            harvest_phase_begin,
            harvest_phase_end,
            season_frac,
            test_region_1s,
            test_years,
            test_pool_frac,
            filter_regions,
            filter_years,
            exclude,
            use_skip_file,
            verbose,
            train_fraction_keep,
            dev_frac,
            scale_factor,
            use_gee_shapefile,
            region_input,
            dataset_year_begin,
            dataset_year_end,
            year_to_skip,
        )
        train_hists, train_ndvi, train_yields, train_keys, train_years, train_locs = (
            list(zip(*train))
        )
        if len(dev) > 0:
            dev_hists, dev_ndvi, dev_yields, dev_keys, dev_years, dev_locs = list(
                zip(*dev)
            )
        if len(test) > 0:
            test_hists, test_ndvi, test_yields, test_keys, test_years, test_locs = list(
                zip(*test)
            )

        if verbose:
            print("Saving datasets to", directory)
            print("Stand by...")

        if "Prediction" in directory:
            np.savez_compressed(pred_hists_save_path, data=train_hists)
            np.savez_compressed(pred_ndvi_save_path, data=train_ndvi)
            np.savez_compressed(pred_yields_save_path, data=train_yields)
            np.savez_compressed(pred_keys_save_path, data=train_keys)
            np.savez_compressed(pred_years_save_path, data=train_years)
            np.savez_compressed(pred_locs_save_path, data=train_locs)
        else:
            np.savez_compressed(train_hists_save_path, data=train_hists)
            np.savez_compressed(train_ndvi_save_path, data=train_ndvi)
            np.savez_compressed(train_yields_save_path, data=train_yields)
            np.savez_compressed(train_keys_save_path, data=train_keys)
            np.savez_compressed(train_years_save_path, data=train_years)
            np.savez_compressed(train_locs_save_path, data=train_locs)

            if len(dev) > 0:
                np.savez_compressed(dev_hists_save_path, data=dev_hists)
                np.savez_compressed(dev_ndvi_save_path, data=dev_ndvi)
                np.savez_compressed(dev_yields_save_path, data=dev_yields)
                np.savez_compressed(dev_keys_save_path, data=dev_keys)
                np.savez_compressed(dev_years_save_path, data=dev_years)
                np.savez_compressed(dev_locs_save_path, data=dev_locs)

            if len(test) > 0:
                np.savez_compressed(test_yields_save_path, data=test_yields)
                assert len(test_ndvi[0].shape) == 1
                np.savez_compressed(test_hists_save_path, data=test_hists)
                np.savez_compressed(test_ndvi_save_path, data=test_ndvi)
                np.savez_compressed(test_keys_save_path, data=test_keys)
                np.savez_compressed(test_years_save_path, data=test_years)
                np.savez_compressed(test_locs_save_path, data=test_locs)

    else:
        train_yields = np.load(train_yields_save_path)["data"]
        print(dev_yields_save_path)
        dev_yields = np.load(dev_yields_save_path)["data"]
        
        test_yields = np.load(test_yields_save_path)["data"]
        if verbose:
            print("Datasets already generated and available in", target_dir)
            print('Rerun with flag "-r" to remake.')

    if verbose:
        if "Prediction" in directory:
            print("Number of prediction examples:", len(train))
        else:
            print("Number of train examples:", len(train))
            print("Number of dev examples:", len(dev))
            print("Number of test examples:", len(test))


def main():
    parser = argparse.ArgumentParser(
        description="Sorts histograms, yield data, and location information into usable datasets."
    )
    parser.add_argument(
        "-k", "--key_file", type=str, help="Path to the key file with input parameters."
    )

    args = parser.parse_args()

    if args.key_file:
        params = read_key_file(args.key_file)
        histogram_directory = params.get("HISTOGRAM_FOLDER")
        target_directory = params.get("DATASET_FOLDER")
        crop_of_interest = params.get("CROP")
        country = params.get("REGION")
        region_input = params.get("REGIONS")
        use_gee_shapefile = params.get("USE_GEE_SHAPEFILE") == "1"
        harvest_year_begin = int(params.get("HARVEST_BEGIN"))
        harvest_year_end = int(params.get("HARVEST_END"))
        harvest_phase_begin = params.get("HARVEST_PHASE_BEGIN")
        harvest_phase_end = params.get("HARVEST_PHASE_END")
        year_to_skip = int(params.get("YEAR_TO_SKIP"))
        dataset_year_begin = int(params.get("DATASET_BEGIN"))
        dataset_year_end = int(params.get("DATASET_END"))
        season_frac = float(params.get("SEASON_FRAC"))
        test_provinces = (
            None
            if params.get("TEST_PROVINCES") == "None"
            else params.get("TEST_PROVINCES")
        )
        test_years = (
            None
            if params.get("TEST_YEARS") == "None"
            else params.get("TEST_YEARS").split(" ")
        )
        test_pool_frac = float(params.get("TEST_POOL_FRAC"))
        filter_provinces = (
            None
            if params.get("FILTER_PROVINCES") == "None"
            else params.get("FILTER_PROVINCES")
        )
        filter_years = (
            None if params.get("FILTER_YEARS") == "None" else params.get("FILTER_YEARS")
        )
        dev_frac_of_train = float(params.get("DEV_FRAC_OF_TRAIN"))
        exclude = params.get("EXCLUDE") == "1"
        remake = params.get("REMAKE") == "1"
        use_skip_file = params.get("USE_SKIP_FILE") == "1"
        verbose = params.get("VERBOSE") == "1"
        train_fraction_keep = float(params.get("TRAIN_FRACTION_KEEP"))
    else:
        print("Key file is required.")
        sys.exit(1)

    if verbose:
        print(
            "This script should only be called in your git folder for this project. Must also have a bucket in ~/ named",
            basename(normpath(GBUCKET)),
            "as well as a local data directory",
            LOCAL_DATA_DIR,
        )
        print()
        print("Crop of interest is", crop_of_interest, "in", country)
        print("Creating data files in", target_directory)
        print("Histograms coming from", histogram_directory)
        print("Using", int(season_frac * 100), "% of each season's data")
        print()
        print(
            "Searching for harvest years", harvest_year_begin, "and", harvest_year_end
        )
        if exclude and test_years is not None:
            print("Only training on years up to", sort_harvest_year_strings(test_years))
        else:
            print("Including all non-test years in training set")
        print()
        print("Proportion of training set kept for training", train_fraction_keep)
        print(
            "Proportion of remaining training examples used for dev set",
            dev_frac_of_train,
        )
        print("Proportion of test candidates used for test", test_pool_frac)
        print()
        if use_skip_file:
            print("Skipping provinces in filtered regions file if file is available")
        else:
            print("Not using filtered regions file")
        print("Filter provinces", filter_provinces)
        print("Filter years", filter_years)
        print()
        print("Test provinces =", test_provinces)
        print("Test years =", test_years)

    #[
    #    os.remove(os.path.join(target_directory, f))
    #    for f in os.listdir(target_directory)
    #    if os.path.isfile(os.path.join(target_directory, f))
    #]
    run(
        histogram_directory,
        target_directory,
        crop_of_interest,
        country,
        harvest_year_begin,
        harvest_year_end,
        harvest_phase_begin,
        harvest_phase_end,
        season_frac,
        test_provinces,
        test_years,
        test_pool_frac,
        filter_provinces,
        filter_years,
        dev_frac_of_train,
        exclude,
        remake,
        use_skip_file,
        verbose,
        train_fraction_keep,
        1,
        use_gee_shapefile,
        region_input,
        dataset_year_begin,
        dataset_year_end,
        year_to_skip,
    )


if __name__ == "__main__":
    main()

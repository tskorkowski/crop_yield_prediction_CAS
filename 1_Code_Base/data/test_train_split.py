from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Normalization


def test_train_split(
    dataset: np.ndarray,
    labels: np.ndarray,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    batch_size: int = 32,
    months: int = 0,
    num_bands: int = 0,
    num_buckets: int = 0,
    **kwargs,
):

    # Split dataset into training, validation and test sets
    # Assumed shape of the dataset: (n_samples, x)

    normalizer = Normalization(axis=-1)

    dataset_train, dataset_test, labels_train, labels_test = train_test_split(
        dataset, labels, test_size=test_size, random_state=42
    )
    dataset_train, dataset_val, labels_train, labels_val = train_test_split(
        dataset_train, labels_train, test_size=validation_size, random_state=42
    )

    tensor_dataset_train = tf.convert_to_tensor(dataset_train, dtype=tf.float32)
    tensor_labels_train = tf.convert_to_tensor(labels_train, dtype=tf.float32)
    tensor_dataset_val = tf.convert_to_tensor(dataset_val, dtype=tf.float32)
    tensor_labels_val = tf.convert_to_tensor(labels_val, dtype=tf.float32)
    tensor_dataset_test = tf.convert_to_tensor(dataset_test, dtype=tf.float32)
    tensor_labels_test = tf.convert_to_tensor(labels_test, dtype=tf.float32)

    tf_dataset_train = tf.data.Dataset.from_tensor_slices(
        (tensor_dataset_train, tensor_labels_train)
    )
    tf_dataset_val = tf.data.Dataset.from_tensor_slices(
        (tensor_dataset_val, tensor_labels_val)
    )
    tf_dataset_test = tf.data.Dataset.from_tensor_slices(
        (tensor_dataset_test, tensor_labels_test)
    )

    tf_dataset_train = tf_dataset_train.batch(
        batch_size, drop_remainder=True
    )  # .shuffle(buffer_size=10000)
    tf_dataset_val = tf_dataset_val.batch(
        batch_size, drop_remainder=True
    )  # .shuffle(buffer_size=10000)
    tf_dataset_test = tf_dataset_test.batch(
        batch_size, drop_remainder=True
    )  # .shuffle(buffer_size=10000)

    normalizer.adapt(tf_dataset_train.map(lambda x, y: x))

    # train_val_size = int((1 - test_size) * dataset.shape[0] / batch_size)
    # val_size = int(train_val_size * validation_size)

    # train_val_dataset = tf_dataset.take(train_val_size)
    # test_dataset = tf_dataset.skip(train_val_size)

    # train_dataset = train_val_dataset.take(train_val_size - val_size)
    # val_dataset = train_val_dataset.skip(train_val_size - val_size)

    # Adapt normalizer on unbatched training data

    train_dataset = tf_dataset_train.map(lambda x, y: (normalizer(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    val_dataset = tf_dataset_val.map(lambda x, y: (normalizer(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    test_dataset = tf_dataset_test.map(lambda x, y: (normalizer(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset, val_dataset, test_dataset


def convert_np_array_to_tf_dataset(np_dataset: Tuple[np.ndarray]) -> tf.data.Dataset:
    """Convert a list of numpy arrays to a list of tf.data.Dataset objects"""

    return tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(np_dataset[0], dtype=tf.float32),
            tf.convert_to_tensor(np_dataset[1], dtype=tf.float32),
        )
    )


def get_common_index(data_sets: List[pd.DataFrame]):
    return reduce(
        lambda x, y: x.intersection(y), [data_set.index for data_set in data_sets]
    )


def test_train_split_multi_modal(
    weather_dataset: str,
    histograms_dataset: str,
    labels: str,
    index: List[str],
    months: List[int],
    training_range: tuple[int, int] = (2017, 2022),
    batch_size: int = 64,
):

    # weather only cover 2017-2022 while sattelite images olso cover 2016
    # For test purpose years 2016 and 2022 are used - since not all modatlities are present models will ber evaluated on 2016 and 2022 separately
    # Training and validation covers the remainding period

    # Sattelite images histgrams are taken to be the main modality
    normalizer = Normalization(axis=-1)

    joined_data = pd.DataFrame()

    data_sets = []
    if histograms_dataset:
        hist_data = (
            pd.DataFrame(
                np.load(histograms_dataset, allow_pickle=True),
            )
            .astype({"fips": str, "year": int})
            .set_index(index)
            .sort_index()
        )
        data_sets.append(hist_data)

    print(
        pd.DataFrame(
            np.load(histograms_dataset, allow_pickle=True),
        ).head()
    )

    print("", "histogram shape:", data_sets[0].shape, sep="\n")

    if weather_dataset:
        weather_data = (
            pd.read_csv(
                weather_dataset,
                dtype={
                    "fips": str,
                    "year": int,
                },
            )
            .drop(columns=["County"])
            .dropna()
            .set_index(index)
            .sort_index()
        )
        data_sets.append(weather_data)

    common_index = get_common_index(data_sets)

    data_sets = [data_set.loc[common_index] for data_set in data_sets]

    datasets_monthly_train = []
    datasets_monthly_test = []
    for data_set in data_sets:
        # one_month_data = data_set[data_set["month"] == months[0]]
        # print(one_month_data.loc[pd.IndexSlice["19043", 2017:], :].sort_index())
        datasets_monthly_train.extend(
            [
                data_set[data_set["month"] == month].loc[
                    pd.IndexSlice[:, training_range[0] : training_range[1]], :
                ]
                for month in months
            ]
        )
        datasets_monthly_test.extend(
            [
                data_set[data_set["month"] == month].loc[
                    pd.IndexSlice[:, training_range[1] :], :
                ]
                for month in months
            ]
        )
    normalizers = {
        f"norm_data_{i}": Normalization(axis=-1)
        for i in range(len(datasets_monthly_train))
    }
    for i, dataset in enumerate(datasets_monthly_train):
        print(dataset.head(6))
    # # Split data by month into separate DataFrames
    # month5_data = joined_data[joined_data["month"] == 5]
    # month7_data = joined_data[joined_data["month"] == 7]
    # month9_data = joined_data[joined_data["month"] == 9]

    # Add suffix to all columns except fips and year
    # for df, suffix in [
    #     (month5_data, "_m5"),
    #     (month7_data, "_m7"),
    #     (month9_data, "_m9"),
    # ]:
    #     df.columns = [
    #         f"{col}{suffix}" if col not in ["fips", "year"] else col
    #         for col in df.columns
    #     ]

    # # Join the DataFrames side by side
    # months_combined_df = month5_data.merge(
    #     month7_data, on=["fips", "year"], how="inner"
    # ).merge(month9_data, on=["fips", "year"], how="inner")

    # labels = pd.DataFrame(np.load(labels, allow_pickle=True), columns=index + ["yield"])

    # dataset = months_combined_df.merge(labels, on=["fips", "year"], how="left")

    # min_year = dataset["year"].min()
    # dataset["year"] = dataset["year"] - min_year

    # dataset = dataset.dropna().drop(columns=["fips"])

    # test_cutoff_bottom = training_range[0] - min_year - 1
    # test_cutoff_top = training_range[1] - min_year

    # test_dataset_wo_weather = dataset[dataset["year"] <= test_cutoff_bottom]
    # test_dataset_sat_weather = dataset[dataset["year"] >= test_cutoff_top]

    # training_dataset = dataset[
    #     (dataset["year"] > test_cutoff_bottom) & (dataset["year"] < test_cutoff_top)
    # ]

    # training_labels = training_dataset.pop("yield")
    # test_labels_wo_weather = test_dataset_wo_weather.pop("yield")
    # test_labels_sat_weather = test_dataset_sat_weather.pop("yield")

    # print("training data shape:", training_dataset.shape)
    # # print("training data head:", training_dataset.head())
    # print("test data wo weather shape:", test_dataset_wo_weather.shape)
    # print("test data shape:", test_dataset_sat_weather.shape)

    # tensor_dataset_train = tf.convert_to_tensor(training_dataset, dtype=tf.float32)
    # tensor_labels_train = tf.convert_to_tensor(training_labels, dtype=tf.float32)
    # tensor_dataset_wo_weather = tf.convert_to_tensor(
    #     test_dataset_wo_weather, dtype=tf.float32
    # )
    # tensor_labels_wo_weather = tf.convert_to_tensor(
    #     test_labels_wo_weather, dtype=tf.float32
    # )
    # tensor_dataset_sat_weather = tf.convert_to_tensor(
    #     test_dataset_sat_weather, dtype=tf.float32
    # )
    # tensor_labels_sat_weather = tf.convert_to_tensor(
    #     test_labels_sat_weather, dtype=tf.float32
    # )

    # tf_dataset_train = tf.data.Dataset.from_tensor_slices(
    #     (tensor_dataset_train, tensor_labels_train)
    # )
    # tf_dataset_test_wo_weather = tf.data.Dataset.from_tensor_slices(
    #     (tensor_dataset_wo_weather, tensor_labels_wo_weather)
    # )
    # tf_dataset_test_weather = tf.data.Dataset.from_tensor_slices(
    #     (tensor_dataset_sat_weather, tensor_labels_sat_weather)
    # )

    # tf_dataset_train = tf_dataset_train.batch(batch_size, drop_remainder=True).shuffle(
    #     buffer_size=10000, seed=42
    # )
    # tf_dataset_test_wo_weather = tf_dataset_test_wo_weather.batch(
    #     batch_size, drop_remainder=True
    # )  # .shuffle(buffer_size=10000)
    # tf_dataset_test_weather = tf_dataset_test_weather.batch(
    #     batch_size, drop_remainder=True
    # )  # .shuffle(buffer_size=10000)

    # normalizer.adapt(tf_dataset_train.map(lambda x, y: x))

    # train_dataset = tf_dataset_train.map(lambda x, y: (normalizer(x), y)).prefetch(
    #     tf.data.AUTOTUNE
    # )

    # test_dataset_wo_weather = tf_dataset_test_wo_weather.map(
    #     lambda x, y: (normalizer(x), y)
    # ).prefetch(tf.data.AUTOTUNE)

    # test_dataset_weather = tf_dataset_test_weather.map(
    #     lambda x, y: (normalizer(x), y)
    # ).prefetch(tf.data.AUTOTUNE)

    # return train_dataset, test_dataset_wo_weather, test_dataset_weather


if __name__ == "__main__":
    from tensorflow.keras.utils import split_dataset

    weather_dataset = r"C:\Users\tskor\Documents\data\WRF-HRRR\split_by_county_and_year\weather-combined.csv"
    histograms_dataset = r"C:\Users\tskor\Documents\data\histograms\histograms_county_year\histograms-combined.npy"
    labels = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_Data\combined_labels_with_fips.npy"

    datasets = test_train_split_multi_modal(
        weather_dataset,
        histograms_dataset,
        labels,
        index=["fips", "year"],
        months=[5, 7, 9],
    )
    # for dataset in datasets:
    #     for features, labels in dataset.take(1):
    #         print("First batch features:", features.numpy()[:5])
    #         print("First batch labels:", labels.numpy()[:5])


# BUG:
# 1. weather data is not workign correctly, at least aggregate file(?)

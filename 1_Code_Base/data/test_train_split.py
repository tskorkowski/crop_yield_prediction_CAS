from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Normalization

# TODO: make sure that fips column in the weather data is correctly formatted as string with leading 0s

SEED = 42
CHANNEL_NORMALIZATION = 1e4
CAT_FEATURES = ["fips"]


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


def create_preproc_head(
    df: pd.DataFrame, cat_columns: List[str]
) -> Tuple[tf.keras.Model, pd.DataFrame]:

    normalizer = Normalization(axis=-1)
    inputs = {}
    numeric_features = []
    numeric_features_dict = {}
    preprocessed = []

    for name, _ in df.items():
        if name in cat_columns:
            dtype = tf.string
        else:
            dtype = tf.float64
            numeric_features.append(name)
            numeric_features_dict[name] = _.to_numpy()

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # numeric_features_dict = {key: value.to_numpy() for key, value in dict(df[numeric_features]).items()}
    normalizer.adapt(
        np.concatenate([value for key, value in sorted(numeric_features_dict.items())])
    )

    # I need a preproc head that will take an input, create on hot encoding for cat values and normalize the numeric values
    # finally concatanate the two and return the tensor for the actual model

    # Q: How does normalization layer work with inputs split into single columns?

    numeric_inputs = [inputs[name] for name in numeric_features]
    numeric_inputs = Concatenate(axis=-1)(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    return (inputs,)


def split_data_by_month(
    data_set: pd.DataFrame, months: List[int]
) -> List[pd.DataFrame]:
    return [data_set[data_set["month"] == month] for month in months]


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

    if histograms_dataset:
        try:
            hist_data = (
                pd.DataFrame(
                    np.load(histograms_dataset, allow_pickle=True),
                )
                .astype({"fips": str, "year": int})
                .set_index(index)
                .sort_index()
            )
        except:
            hist_data = (
                pd.read_csv(
                    histograms_dataset,
                    dtype={
                        "fips": str,
                        "year": int,
                    },
                )
                .set_index(index)
                .sort_index()
            )
        print("histogram data shape:", hist_data.shape)

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
        print("weather data shape:", weather_data.shape)

    common_index = get_common_index([hist_data, weather_data])

    hist_data = hist_data.loc[common_index]
    weather_data = weather_data.loc[common_index]

    hist_train = hist_data.loc[
        pd.IndexSlice[:, training_range[0] : training_range[1]], :
    ].reset_index()
    hist_test = hist_data.loc[pd.IndexSlice[:, training_range[1] :], :].reset_index()

    weather_train = weather_data.loc[
        pd.IndexSlice[:, training_range[0] : training_range[1]], :
    ].reset_index()
    weather_test = weather_data.loc[
        pd.IndexSlice[:, training_range[1] :], :
    ].reset_index()

    hist_train_monthly = split_data_by_month(hist_train, months)
    hist_test_monthly = split_data_by_month(hist_test, months)

    weather_train_monthly = split_data_by_month(weather_train, months)
    weather_test_monthly = split_data_by_month(weather_test, months)

    return (hist_train_monthly, weather_train_monthly), (
        hist_test_monthly,
        weather_test_monthly,
    )


if __name__ == "__main__":
    from tensorflow.keras.utils import split_dataset

    test_data = True

    if test_data:
        weather_dataset = r"4_Data_Sample\weather_data-adams-2017-CO-8001.csv"
        histograms_dataset = r"4_Data_Sample\histogram-08001-2017_reduced.csv"
        labels = r"4_Data_Sample\combined_labels_with_fips.npy"
    else:
        weather_dataset = r"C:\Users\tskor\Documents\data\WRF-HRRR\split_by_county_and_year\weather-combined.csv"
        histograms_dataset = r"C:\Users\tskor\Documents\data\histograms\histograms_county_year\histograms-combined.npy"
        labels = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_Data\combined_labels_with_fips.npy"

    datasets_monthly_train, datasets_monthly_test = test_train_split_multi_modal(
        weather_dataset,
        histograms_dataset,
        labels,
        index=["fips", "year"],
        months=[5, 7, 9],
    )

    histograms_train, weather_train = datasets_monthly_train
    inputs = create_preproc_head(df=histograms_train[0], cat_columns=CAT_FEATURES)
    print(inputs)
    # print(len(inputs), '\n\n', inputs)
    # for dataset in datasets:
    #     for features, labels in dataset.take(1):
    #         print("First batch features:", features.numpy()[:5])
    #         print("First batch labels:", labels.numpy()[:5])


# BUG:
# 1. weather data is not workign correctly, at least aggregate file(?)

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


if __name__ == "__main__":
    dataset_path = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline\test_data\dataset_nan_map_True_norm_True_60_buckets_12_bands_60_dataset.npy"
    labels_path = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline\test_data\dataset_nan_map_True_norm_True_60_buckets_12_bands_60_labels.npy"
    dataset = np.load(dataset_path)
    labels = np.load(labels_path)
    config = {
        "months": 3,
        "num_bands": 12,
        "num_buckets": 60,
    }
    dataset = np.concatenate(
        [dataset[:, 0, :], dataset[:, 1, :], dataset[:, 2, :]], axis=-1
    )
    train_dataset, val_dataset, test_dataset = test_train_split(
        dataset, labels, **config
    )


def test_train_split_multi_modal(
    weather_dataset: str,
    histograms_dataset: str,
    labels: str,
    required_shape: tuple,
    training_range: tuple[int, int] = (2017, 2022),
):

    # weather only cover 2017-2022 while sattelite images olso cover 2016
    # For test purpose years 2016 and 2022 are used - since not all modatlities are present models will ber evaluated on 2016 and 2022 separately
    # Training and validation covers the remainding period

    # Sattekite images histgrams are taken to be the main modality

    # hist_df = pl.DataFrame({
    #     'histogram': histograms_dataset[:, :-2].tolist(),  # Store features as list
    #     'year': histograms_dataset[:, -2].astype(int),
    #     'fips': histograms_dataset[:, -1],
    #     'source': 'histogram'
    # })

    # weather_df = pl.DataFrame({
    #     'weather': weather_dataset[:, :-2].tolist(),  # Store features as list
    #     'year': weather_dataset[:, -2].astype(int),
    #     'fips': weather_dataset[:, -1],
    #     'source': 'weather'
    # })

    # # Join on fips and year
    # joined_df = hist_df.join(
    #     weather_df,
    #     on=['fips', 'year'],
    #     how='inner'  # or 'outer', 'left', 'right' depending on your needs
    # )

    # Convert to pandas for easier joining

    hist_df = pd.DataFrame(np.load(histograms_dataset))
    weather_data = pd.read_csv(weather_dataset)

    # Join using multiple keys
    joined_data = hist_df.merge(weather_data, on=["fips", "year", "month"], how="left")
    return train_dataset, val_dataset, test_dataset

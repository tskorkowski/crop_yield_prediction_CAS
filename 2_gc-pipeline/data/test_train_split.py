import numpy as np
import tensorflow as tf
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

    tensor_dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
    tensor_labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    tf_dataset = tf.data.Dataset.from_tensor_slices((tensor_dataset, tensor_labels))
    tf_dataset = tf_dataset.shuffle(buffer_size=10000).batch(
        batch_size, drop_remainder=True
    )

    normalizer.adapt(tf_dataset.map(lambda x, y: x))

    train_val_size = int((1 - test_size) * dataset.shape[0] / batch_size)
    val_size = int(train_val_size * validation_size)

    train_val_dataset = tf_dataset.take(train_val_size)
    test_dataset = tf_dataset.skip(train_val_size)

    train_dataset = train_val_dataset.take(train_val_size - val_size)
    val_dataset = train_val_dataset.skip(train_val_size - val_size)

    # Adapt normalizer on unbatched training data

    train_dataset = train_dataset.map(lambda x, y: (normalizer(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.map(lambda x, y: (normalizer(x), y)).prefetch(
        tf.data.AUTOTUNE
    )

    test_dataset = test_dataset.map(lambda x, y: (normalizer(x), y)).prefetch(
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

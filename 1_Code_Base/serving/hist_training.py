import datetime
import glob
import io
import itertools
import logging
import os
import random
import sys
import tempfile

import google.auth
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import randomname
import tensorflow as tf
import wandb
from google.cloud import storage
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    TimeDistributed,
)
from serving.constants import (
    BUCKET,
    HEADER_PATH,
    HIST_BINS_LIST,
    HIST_DEST_PREFIX,
    LABELS_PATH,
    MAP_NAN,
    MONTHS,
    NORMALIZE,
    NUM_BINS,
    PIX_COUNT,
    SCALE,
    SELECTED_BANDS,
)
from serving.data import get_labels
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# def set_seeds(seed=42):
#     # Set seeds for Python's random module
#     random.seed(seed)

#     # Set seed for NumPy
#     np.random.seed(seed)

#     # Set seed for TensorFlow
#     tf.random.set_seed(seed)


# Define the LSTM model
class LstmModel(keras.Model):
    def __init__(
        self,
        input_shape,
        lstm_layers=3,
        no_units=3,
        output_units=1,
        dropout_rate=0.2,
        val_size=10,
        kernel_initializer=tf.keras.initializers.RandomNormal(),
    ):
        super(LstmModel, self).__init__()
        self.lstm_layers = lstm_layers
        self.no_units = no_units
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.val_size = val_size
        self.norm = normalizer

        self.input_layer = Input(shape=input_shape)
        self.batch_norm = BatchNormalization()

        # Define LSTM and Dense layers
        self.lstm_layers_list = []
        for i in range(lstm_layers):
            self.lstm_layers_list.append(
                LSTM(
                    units=no_units,
                    return_sequences=(i < lstm_layers - 1),
                    kernel_initializer=kernel_initializer,
                )
            )

            if i < lstm_layers - 1:
                self.lstm_layers_list.append(Dropout(rate=dropout_rate))

        self.dense = Dense(units=output_units, use_bias=False)

        self.job_name = randomname.get_name(adj=("emotions",), noun=("food"))

    def get_config(self):
        return {
            "lstm_layers": self.lstm_layers,
            "no_units": self.no_units,
            "val_size": self.val_size,
            "ouput_units": self.output_units,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer,
        }

    @classmethod
    def from_config(cls, config):
        # Convert the serialized initializer back to a TF initializer
        config["kernel_initializer"] = tf.keras.initializers.deserialize(
            config["kernel_initializer"]
        )
        return cls(**config)

    @tf.function
    def call(self, inputs, training=False):
        # LSTM layers

        x = inputs
        # x = self.batch_norm(x, training=training)

        for lstm_layer in self.lstm_layers_list:
            x = lstm_layer(x)

        outputs = self.dense(x)

        return outputs

    def summary(self):
        super(LstmModel, self).summary()

    def compile(
        self, optimizer="adam", loss="mse", metrics=["mae"], learning_rate=0.001
    ):
        self.learning_rate = learning_rate

        # Dictionary to map optimizer names to their classes
        optimizers = {"adam": Adam, "nadam": Nadam, "rms": RMSprop}

        # Get the optimizer class from the dictionary
        optimizer_class = optimizers.get(optimizer.lower(), Adam)

        # Instantiate the optimizer with the specified learning rate
        optimizer_instance = optimizer_class(learning_rate=learning_rate, clipnorm=1.0)

        # Compile the model with the chosen optimizer, loss, and metrics
        tf.config.run_functions_eagerly(True)

        if loss != "mse" and "mse" not in metrics:
            metrics.append("mse")

        super(LstmModel, self).compile(
            optimizer=optimizer_instance, loss=loss, metrics=metrics
        )

    def fit(self, dataset, epochs=10):

        # Shuffle and batch the dataset
        # dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)

        # Split the dataset
        val_size = self.val_size
        val_dataset = dataset.take(val_size)
        train_dataset = dataset.skip(val_size)

        # Setup tensorboard
        model_name = (
            f"{NUM_BINS}_buckets_{len(HIST_BINS_LIST)}"
            + "-"
            + self.job_name
            + "-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        log_dir = "gs://vgnn/tensorboard-artifacts/logs/fit/" + model_name
        if not os.path.exists(os.path.dirname(log_dir)):
            os.makedirs(os.path.dirname(log_dir))

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        # Train the model
        history = super(LstmModel, self).fit(
            train_dataset,
            epochs=epochs,
            # batch_size=batch_size,
            validation_data=val_dataset,
            callbacks=[early_stopping, WandbMetricsLogger()],
        )

        responses_train = np.concatenate(
            [response.numpy() for _, response in train_dataset], axis=0
        )
        mean_response_train = np.mean(responses_train)

        responses_val = np.concatenate(
            [response.numpy() for _, response in val_dataset], axis=0
        )

        if val_size == 0:
            self.naive_loss = np.nan
        else:
            self.naive_loss = pen_low_lenient_high_loss(
                responses_val, mean_response_train
            )

        # Plot training progress
        plot_training_progress(history, self.naive_loss)

        # Evaluate the model
        loss = self.evaluate(val_dataset)

        self.save(f"gs://vgnn/models/{model_name}.h5")

        return history


class TimeDependentDenseLstmModel(LstmModel):
    def __init__(
        self,
        input_shape,
        normalizer,
        lstm_layers=3,
        no_units=3,
        dense_layers_per_step=3,
        output_units=1,
        dropout_rate=0.2,
        val_size=10,
        kernel_initializer=tf.keras.initializers.RandomNormal(),
    ):
        super(TimeDependentDenseLstmModel, self).__init__(
            input_shape,
            normalizer,
            lstm_layers,
            no_units,
            output_units,
            dropout_rate,
            val_size,
            kernel_initializer,
        )

        self.lstm_layers = lstm_layers
        self.dense_layers_per_step = dense_layers_per_step
        self.no_units = no_units
        self.val_size = val_size
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Dense layer to process each time step using TimeDistributed
        self.time_distributed_dense = []
        # Create the dense layers that will be applied to each time step
        units = input_shape[-1] // 2
        for _ in range(dense_layers_per_step):
            self.time_distributed_dense.append(
                TimeDistributed(
                    Dense(units=units, activation="relu", bias_regularizer=l2(0.01))
                )
            )
            # units //= 2  # Halve units for each subsequent dense layer

    def get_config(self):
        config = super(TimeDependentDenseLstmModel, self).get_config()
        config.update(
            {
                "dense_layers_per_step": self.dense_layers_per_step,
            }
        )
        return config

    @tf.function
    def call(self, inputs, training=False):
        # Process each time step with a corresponding dense layer

        x = inputs
        # x = self.batch_norm(x, training=training)
        for dense_layer in self.time_distributed_dense:
            x = dense_layer(x)

        # Feed into LSTM layers
        for lstm_layer in self.lstm_layers_list:
            x = lstm_layer(x)

        outputs = self.dense(x)  # Final dense layer for output
        return outputs


def plot_training_progress(history, naive_loss=None):
    epochs = len(history.history["loss"])

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    if naive_loss is not None:
        epochs = len(history.history["loss"])
        plt.hlines(
            naive_loss,
            label="Naive Loss",
            xmin=0,
            xmax=epochs - 1,
            colors="r",
            linestyles="dashed",
        )

    plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(
    #     history.history["val_loss"], label="Validation Loss"    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.tight_layout()
    plt.show()


def combine_crop_data(path, save=False):

    # header:
    header = [
        "commodity_desc",
        "reference_period_desc",
        "year",
        "state_name",
        "county_name",
        "YIELD, MEASURED IN BU / ACRE",
    ]

    # Use glob to get all the csv files in the directory and its subdirectories
    all_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

    # Read each file into a dataframe
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, usecols=header, index_col=None, header=0)
        df["source_file"] = os.path.basename(filename)
        df_list.append(df)

    # Combine all dataframes
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    combined_df.rename(columns={"YIELD, MEASURED IN BU / ACRE": "target"}, inplace=True)

    # Perform any necessary data cleaning or transformation here
    # For example:
    # combined_df['date'] = pd.to_datetime(combined_df['date'])
    # combined_df = combined_df.dropna()

    # Convert to numpy arrays
    combined_labels = combined_df.to_numpy()

    if save:
        # Save as numpy arrays
        np.save("labels_combined.npy", combined_labels)

        # Optionally, save column names for reference
        np.save("labels_header.npy", combined_df.columns.to_numpy())

    return combined_df


def load_model_from_gcs(model_name, bucket_name="vgnn"):
    # Construct the GCS path
    gcs_model_path = f"gs://{bucket_name}/models/{model_name}"

    # Load the model directly from the GCS path
    model = tf.keras.models.load_model(gcs_model_path)

    return model


def pen_low_lenient_high_loss(
    y_true, y_pred, low_yield_threshold=80.0, high_yield_threshold=200.0
):
    """
    Custom loss function that focuses on recognizing low crop yield.

    Args:
        y_true: Tensor of true crop yield values.
        y_pred: Tensor of predicted crop yield values.
        low_yield_threshold: Threshold below which yields are considered low.

    Returns:
        loss: Computed loss value.
    """
    # Calculate the absolute error
    squared_error = tf.square(y_true - y_pred)

    # Define weights based on whether the true yield is below the threshold
    weights = tf.where(
        y_true < low_yield_threshold, 5.0, 1.0
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true > high_yield_threshold, 0.5, weights
    )  # Lenient penalty for high yields

    # Compute weighted absolute error
    weighted_squared_error = weights * squared_error

    # You can choose to use Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    loss = tf.reduce_mean(weighted_squared_error)

    return loss


def pen_low_loss(y_true, y_pred, low_yield_threshold=80.0, high_yield_threshold=200.0):
    """
    Custom loss function that focuses on recognizing low crop yield.

    Args:
        y_true: Tensor of true crop yield values.
        y_pred: Tensor of predicted crop yield values.
        low_yield_threshold: Threshold below which yields are considered low.

    Returns:
        loss: Computed loss value.
    """
    # Calculate the absolute error
    squared_error = tf.square(y_true - y_pred)

    # Define weights based on whether the true yield is below the threshold
    weights = tf.where(
        y_true < low_yield_threshold / 2, 8.0, 5.0
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true >= low_yield_threshold, 1, weights
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true > high_yield_threshold, 0.9, weights
    )  # Lenient penalty for high yields

    # Compute weighted absolute error
    weighted_squared_error = weights * squared_error

    # You can choose to use Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    loss = tf.reduce_mean(weighted_squared_error)

    return loss


def save_dataset_to_gcp(dataset, labels, bucket_name="vgnn", directory="dataset"):
    # Initialize GCP client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    buffer = io.BytesIO()
    np.save(buffer, labels)
    buffer.seek(0)

    destination_blob_name = f"{directory}/labels.npy"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print("Labels uploaded to GCS successfully!")

    buffer = io.BytesIO()
    np.save(buffer, dataset)
    buffer.seek(0)

    destination_blob_name = f"{directory}/dataset.npy"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print("Dataset uploaded to GCS successfully!")

    print(f"Dataset saved to gs://{bucket_name}/{directory}")


def load_dataset_from_gcp(bucket_name="vgnn", directory="dataset"):
    # Initialize GCP client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{directory}/labels.npy"
    blob = bucket.blob(blob_name)

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    labels = np.load(buffer)

    blob_name = f"{directory}/dataset.npy"
    blob = bucket.blob(blob_name)

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    histogram = np.load(buffer)

    # Create TensorFlow tensors
    histogram_tensor = tf.convert_to_tensor(histogram, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((histogram_tensor, label_tensor))

    return dataset, histogram.shape


def create_hist_dataset(
    hist_list: list,
    labels_path: str = LABELS_PATH,
    header_path: str = HEADER_PATH,
    num_bins=NUM_BINS,
    num_bands=len(HIST_BINS_LIST),
    map_nan=MAP_NAN,
    normalize=NORMALIZE,
) -> tf.data.Dataset:

    logging.basicConfig(
        filename=f"crate_dataset.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    def combine_lists(a: list, b: list):
        for array in b:
            assert isinstance(
                array, np.ndarray
            ), f"type of histogram is not numpy nd array: {type(list)}"
        return a + b

    client = storage.Client()
    bucket = client.get_bucket(BUCKET)

    hist_name_base = f"{HIST_DEST_PREFIX}"
    histograms = []
    labels = []

    label_df = get_labels(labels_path, header_path)

    zeros = 0
    skip = 0
    for combination in hist_list:
        county, fips, year = combination

        try:
            label = float(
                label_df.loc[
                    (label_df["county_name"] == county.upper())
                    & (label_df["year"] == year)
                    & (label_df["state_ansi"] == fips),
                    "target",
                ].iloc[0]
            )
        except IndexError:
            logging.info(
                "County {} in {} does not exist in ground truth data. This histogram will be ignored".format(
                    county.upper(), year
                )
            )
            skip += 1
            continue

        zero_count = 0
        hist_by_year = []
        for month in MONTHS:
            file_name = f"{hist_name_base}/nan_map_{map_nan}/norm_{normalize}/{num_bins}_buckets_{num_bands}_bands/{SCALE}/{county.capitalize()}_{fips}/{year}/{month}-{month+1}.npy"
            hist_blob = bucket.blob(file_name)

            if hist_blob.exists():
                content = hist_blob.download_as_bytes()
                binary_data = io.BytesIO(content)
                array = np.load(binary_data)
            else:
                logging.info(
                    "County {}_{} in {} and month {} does not exist in the histogram set. Zero will be used instead".format(
                        county.upper(), fips, year, month
                    )
                )
                array = np.zeros(num_bins * num_bands)
                zero_count += 1

            hist_by_year.append(array)

        if zero_count > 1:
            skip += 1
            continue

        zeros += zero_count
        histograms = combine_lists(histograms, hist_by_year)
        labels.append(label)

    # Convert lists to numpy arrays
    histograms_np = np.array(histograms, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)

    # Reshape the histograms array
    reshaped_histograms = histograms_np.reshape(-1, len(MONTHS), num_bins * num_bands)

    # Ensure labels are 2D
    labels_np = labels_np.reshape(-1, 1)

    print(f"Reshaped histograms shape: {reshaped_histograms.shape}")
    print(f"Labels shape: {labels_np.shape}")
    print(f"Number of filtered combinations: {skip}")
    print(f"Number of missing histograms replaced by zeros: {zeros}")

    assert (
        reshaped_histograms.shape[0] == len(hist_list) - skip
    ), "Something went wrong when aggregating training data"

    # Create TensorFlow tensors
    histogram_tensor = tf.convert_to_tensor(reshaped_histograms, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels_np, dtype=tf.float32)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((histogram_tensor, label_tensor))

    return dataset, reshaped_histograms.shape, reshaped_histograms, labels_np


def train_wandb():
    try:
        wandb.init()

        map_nan = True
        normalize = True
        # Get dataset for testing
        num_bins, num_bands = wandb.config.num_bins_num_bands
        directory = f"dataset/nan_map_{map_nan}/norm_{normalize}/{num_bins}_buckets_{num_bands}_bands/{SCALE}"
        dataset, dataset_shape = load_dataset_from_gcp(directory=directory)
        input_shape = dataset_shape[1:]

        print(
            f"Dataset shape: {dataset_shape}", f"Input_shpae: {input_shape}", sep="\n"
        )

        # Train test split
        test_train_split = 0.8
        train_size = int(dataset_shape[0] * test_train_split / wandb.config.batch_size)
        val_size = int(train_size * (1 - test_train_split))

        dataset_batched = dataset.batch(wandb.config.batch_size, drop_remainder=True)
        test_dataset = dataset_batched.skip(train_size)
        train_dataset = dataset_batched.take(train_size)

        train_dataset = train_dataset.shuffle(buffer_size=10000)
        print(
            f"Train size: {train_size} [batches]\nValidation size: {val_size} [batches]"
        )
        print("Data sets have been setup")

        lstm_layers, lstm_units = wandb.config.lstm_layers_units

        model = TimeDependentDenseLstmModel(
            input_shape=input_shape,
            lstm_layers=lstm_layers,
            dense_layers_per_step=wandb.config.dense_layers_per_step,
            no_units=lstm_units,
            val_size=val_size,
            dropout_rate=wandb.config.dropout_rate,
            kernel_initializer=initializers[wandb.config.kernel_initializer],
        )
        print("Model has been defined")

        model.compile(
            optimizer=wandb.config.optimizer,
            learning_rate=wandb.config.lr,
            loss=pen_low_lenient_high_loss,
        )

        print("Model compiled")

        history = model.fit(train_dataset, epochs=wandb.config.epochs)

    except ValueError as e:
        print(f"Value error in configuration or dataset: {e}")
    except tf.errors.InvalidArgumentError as e:
        print(f"TensorFlow-specific error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        wandb.alert(title="Training Error", text=str(e))
    finally:
        wandb.finish()


def train_model_data(
    model_name: str,
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    test_train_split: float = 0.8,
):

    model = load_model_from_gcs(model_name=model_name)
    data, hist_shape = load_dataset_from_gcp(data_dir)

    train_size = int(hist_shape[0] * test_train_split / batch_size)
    val_size = int(train_size * (1 - test_train_split))

    dataset_batched = dataset.batch(batch_size, drop_remainder=True)
    test_dataset = dataset_batched.skip(train_size)
    train_dataset = dataset_batched.take(train_size)

    history = model.fit(train_dataset, epochs=epochs)

    Print("## Test set evaluation ##")
    model.evaluate(test_dataset)

    return history


if __name__ == "__main__":
    pass

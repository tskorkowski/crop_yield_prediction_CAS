import glob
import io
import itertools
import logging
import os
import random
import sys

import google.auth
import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from keras.layers import LSTM, Dense, Dropout, Input
from mlflow.models import infer_signature
from serving.constants import (
    BUCKET,
    HEADER_PATH,
    HIST_DEST_PREFIX,
    LABELS_PATH,
    MONTHS,
    NUM_BANDS,
    NUM_BINS,
    PIX_COUNT,
    SCALE,
)
from serving.data import get_labels
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop


def set_seeds(seed=42):
    # Set seeds for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for TensorFlow
    tf.random.set_seed(seed)


# Define the LSTM model
class LstmModel(keras.Model):
    def __init__(
        self,
        input_shape,
        lstm_layers=3,
        no_units=3,
        output_units=1,
        dropout_rate=0.2,
        mean_response=0,
    ):
        super(LstmModel, self).__init__()
        self.lstm_layers = lstm_layers
        self.no_units = no_units
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.mean_response = mean_response

        # Define LSTM and Dense layers
        self.lstm_layers_list = [
            LSTM(
                units=no_units,
                return_sequences=(i < lstm_layers - 1),
                kernel_initializer="zeros",
                recurrent_initializer="zeros",
                bias_initializer="zeros",
            )
            for i in range(lstm_layers)
        ]
        self.dense = Dense(
            units=output_units,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(mean_response),
        )

    @tf.function
    def call(self, inputs, training=False):
        # LSTM layers
        x = inputs

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
        optimizer_instance = optimizer_class(learning_rate=learning_rate)

        # Compile the model with the chosen optimizer, loss, and metrics
        tf.config.run_functions_eagerly(True)
        super(LstmModel, self).compile(
            optimizer=optimizer_instance, loss=loss, metrics=metrics
        )

    def fit(self, dataset, epochs=10, batch_size=32):

        class MyModel(mlflow.pyfunc.PythonModel):
            def predict(self, ctx, model_input, params):
                return list(params.values())

        # Set up MLflow experiment
        mlflow.set_experiment("LSTM_Experiment")

        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)

        # Split the dataset
        val_size = 7
        val_dataset = dataset.take(val_size)
        train_dataset = dataset.skip(val_size)

        responses_train = np.concatenate(
            [response.numpy() for _, response in train_dataset], axis=0
        )
        mean_response_train = np.mean(responses_train)

        responses_val = np.concatenate(
            [response.numpy() for _, response in val_dataset], axis=0
        )
        self.naive_loss = np.mean(
            (responses_val - mean_response_train) ** 2
        )  # Mean Squared Error

        # Start MLflow run
        with mlflow.start_run():
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )

            # Create a history callback
            history_callback = tf.keras.callbacks.History()

            # Train the model
            history = super(LstmModel, self).fit(
                train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_dataset,
            )  # callbacks=[early_stopping, history_callback]

            # Log model and parameters to MLflow
            # Create an example input
            for features, _ in dataset.take(1):
                example_input = features.numpy()
            signature = infer_signature(["input"], ["output"])
            model_info = mlflow.pyfunc.log_model(
                python_model=MyModel(), artifact_path="my_model", signature=signature
            )
            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lstm_layers", self.lstm_layers)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_metric("naive_loss", self.naive_loss)
            # Plot training progress
            plot_training_progress(history, self.naive_loss)

            # Evaluate the model
            loss = super(LstmModel, self).evaluate(val_dataset)
            mlflow.log_metric("val_loss", loss)

        return history


class TimeDependentDenseLstmModel(LstmModel):
    def __init__(
        self,
        input_shape,
        lstm_layers=3,
        no_units=3,
        dense_layers_per_step=3,
        output_units=1,
        dropout_rate=0.2,
        mean_response=0,
    ):
        super(TimeDependentDenseLstmModel, self).__init__(
            input_shape,
            lstm_layers,
            no_units,
            output_units,
            dropout_rate,
            mean_response,
        )

        # Add dense layers to process each time step before LSTM
        self.dense_layers_per_time_step = []
        for _ in range(input_shape[1]):
            dense_layers = []
            units = input_shape[-1] // 4
            for _ in range(dense_layers_per_step):
                dense_layers.append(Dense(units=units, activation="relu"))
                units //= 4
            self.dense_layers_per_time_step.append(dense_layers)

    @tf.function
    def call(self, inputs, training=False):
        # Process each time step with a corresponding dense layer
        x = tf.unstack(inputs, axis=1)  # Unstack time steps
        processed_time_steps = []

        for t, dense_layers in zip(x, self.dense_layers_per_time_step):
            for dense_layer in dense_layers:
                t = dense_layer(t)
            processed_time_steps.append(t)

        # Stack back the time steps
        x = tf.stack(processed_time_steps, axis=1)

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
        plt.hlines(naive_loss, label="Naive Loss", xmin=0, xmax=epochs - 1)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(
        history.history["val_loss"], label="Validation Loss", c="r", linestyles="dashed"
    )

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


def create_hist_dataset(
    hist_list: list,
    labels_path: str = LABELS_PATH,
    header_path: str = HEADER_PATH,
    num_bins=NUM_BINS,
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
            file_name = f"{hist_name_base}/{num_bins}_buckets/{SCALE}/{county.capitalize()}_{fips}/{year}/{month}-{month+1}.npy"
            hist_blob = bucket.blob(file_name)

            if hist_blob.exists():
                content = hist_blob.download_as_bytes()
                binary_data = io.BytesIO(content)
                array = np.load(binary_data) // PIX_COUNT
            else:
                logging.info(
                    "County {}_{} in {} and month {} does not exist in the histogram set. Zero will be used instead".format(
                        county.upper(), fips, year, month
                    )
                )
                array = np.zeros(num_bins * NUM_BANDS)
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
    reshaped_histograms = histograms_np.reshape(-1, len(MONTHS), num_bins * NUM_BANDS)

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

    return dataset, reshaped_histograms.shape[1:]


def create_data_sample():
    # Example: X is a list of NumPy arrays, y is a list of corresponding labels
    X = [
        np.random.rand(3, 416),
        np.random.rand(3, 416),
        np.random.rand(3, 416),
    ]  # Input features (e.g., 3 timesteps, 416 features)
    y = [165.5, 179.7, 172.5]  # Corresponding labels

    # Step 1: Convert the lists to NumPy arrays (optional but recommended for consistency)
    X = np.array(X)
    y = np.array(y)

    # Step 2: Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # Step 3: Shuffle and batch the dataset
    batch_size = 2  # Set the batch size
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)

    # Step 4: Inspect the dataset
    for data_point, label in dataset.take(1):
        print("Data point shape (after batching):", data_point.shape)
        print("Label shape (after batching):", label.shape)

    return dataset


def save_dataset_to_gcp(
    dataset, bucket_name="vgnn", file_name="hist_dataset_medium.tfrecords"
):
    # Initialize GCP client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a local temporary file
    local_file_name = "temp_" + file_name

    # Write the dataset to the local file
    with tf.io.TFRecordWriter(local_file_name) as writer:
        for features, label in dataset:
            example = serialize_example(features, label)
            writer.write(example)

    # Upload the local file to GCS
    blob = bucket.blob(f"dataset/{file_name}")
    blob.upload_from_filename(local_file_name)

    # Remove the local temporary file
    os.remove(local_file_name)

    print(f"Dataset saved to gs://{bucket_name}/dataset/{file_name}")


def parse_tfrecord_fn(example_proto):
    # Define the features dictionary that matches the structure used when saving
    feature_description = {
        "feature": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.float32),
    }

    # Parse the input tf.Example proto using the feature description
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the feature from the parsed example
    feature = tf.io.parse_tensor(parsed_features["feature"], out_type=tf.float32)
    label = parsed_features["label"]

    return feature, label


def load_dataset_from_gcp(
    bucket_name="vgnn", file_name="hist_dataset_medium.tfrecords"
):
    # Construct the full GCS path
    gcs_path = f"gs://{bucket_name}/dataset/{file_name}"

    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(gcs_path)

    # Parse the TFRecords
    parsed_dataset = dataset.map(parse_tfrecord_fn)

    return parsed_dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "feature": _bytes_feature(tf.io.serialize_tensor(features)),
        "label": _float_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def save_dataset_to_gcp(
    dataset, bucket_name="vgnn", file_name="hist_dataset_medium.tfrecords"
):
    # Initialize GCP client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a local temporary file
    local_file_name = "temp_" + file_name

    # Write the dataset to the local file
    with tf.io.TFRecordWriter(local_file_name) as writer:
        for features, label in dataset:
            example = serialize_example(features, label)
            writer.write(example)

    # Upload the local file to GCS
    blob = bucket.blob(f"dataset/{file_name}")
    blob.upload_from_filename(local_file_name)

    # Remove the local temporary file
    os.remove(local_file_name)

    print(f"Dataset saved to gs://{bucket_name}/dataset/{file_name}")


def parse_tfrecord_fn(example_proto):
    # Define the features dictionary that matches the structure used when saving
    feature_description = {
        "feature": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([1], tf.float32),
    }

    # Parse the input tf.Example proto using the feature description
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the feature from the parsed example
    feature = tf.io.parse_tensor(parsed_features["feature"], out_type=tf.float32)
    label = parsed_features["label"]

    return feature, label


def load_dataset_from_gcp(
    bucket_name="vgnn", file_name="hist_dataset_medium.tfrecords"
):
    # Construct the full GCS path
    gcs_path = f"gs://{bucket_name}/dataset/{file_name}"

    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(gcs_path)

    # Parse the TFRecords
    parsed_dataset = dataset.map(parse_tfrecord_fn)

    return parsed_dataset


# Function to check shapes
def check_dataset_shapes(dataset, num_samples_to_check=3):
    for i, (features, label) in enumerate(dataset.take(num_samples_to_check)):
        print(f"Sample {i+1}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Label shape: {label.shape}")
        if i == 0:
            first_shape = features.shape
        else:
            if features.shape != first_shape:
                print("Warning: Inconsistent feature shape detected!")
                break


if __name__ == "__main__":
    dataset = load_dataset_from_gcp()

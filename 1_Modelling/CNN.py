import argparse
import logging
import os
import random
import sys

import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

logging.debug("Imports completed")


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Define the CNN model
class CnnModel(keras.Model):
    def __init__(
        self,
        input_shape,
        conv_layers=3,
        filters=32,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        dense_units=64,
        output_units=1,
        dropout_rate=0.4,
    ):
        super(CnnModel, self).__init__()
        self.conv_layers = []

        self.input_layer = Input(shape=input_shape)

        for i in range(conv_layers):
            self.conv_layers.append(
                Conv2D(filters * (2**i), kernel_size, activation="relu", padding="same")
            )
            self.conv_layers.append(MaxPooling2D(pool_size=pool_size))
            self.conv_layers.append(Dropout(dropout_rate))

        self.flatten = Flatten()
        self.dense = Dense(dense_units, activation="relu")
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(output_units)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)


# Create TensorFlow dataset
def create_dataset(X, y, batch_size=32, shuffle=True):
    """
    X original shape (n, height, time, channels) where:
    n - number of samples
    height, time - dimensions for 2D convolutions
    channels - number of channels (e.g., spectral bands)
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Main training loop
def train_and_evaluate(
    model, train_dataset, val_dataset, epochs=10000, learning_rate=0.001, patience=10
):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    class MLflowCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metrics(logs, step=epoch)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping, MLflowCallback()],
        verbose=1,
    )

    return model, history


def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()
    return params


def load_data(file_path, dtype=np.float32):
    try:
        data = np.load(file_path)["data"]
        return data.astype(dtype)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def plot_loss(history):
    plt.ion()
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


# Main execution
if __name__ == "__main__":
    logging.debug("Entering main block")
    set_seeds()

    parser = argparse.ArgumentParser(description="Trains CNN architecture.")
    parser.add_argument(
        "-k", "--key_file", type=str, help="Path to the key file with input parameters."
    )
    args = parser.parse_args()

    if args.key_file:
        params = read_key_file(args.key_file)
        dataset_source_dir = params.get("DATASET_FOLDER")
        # Add other necessary parameters here
    else:
        print("Key file is required.")
        sys.exit(1)

    # Load data
    train_data = load_data(
        os.path.join(dataset_source_dir, "train_hists.npz"), dtype=np.float32
    )
    train_labels = load_data(
        os.path.join(dataset_source_dir, "train_yields.npz"), dtype=np.float32
    )
    test_data = load_data(
        os.path.join(dataset_source_dir, "test_hists.npz"), dtype=np.float32
    )
    test_labels = load_data(
        os.path.join(dataset_source_dir, "test_yields.npz"), dtype=np.float32
    )

    train_dataset = create_dataset(train_data, train_labels)
    val_dataset = create_dataset(test_data, test_labels)

    # Initialize and train the model
    input_shape = train_data.shape[1:]  # (t, bin, band)
    model = CnnModel(input_shape)
    trained_model, history = train_and_evaluate(model, train_dataset, val_dataset)

    # Save the model
    trained_model.save("cnn_model.keras")

    plot_loss(history)

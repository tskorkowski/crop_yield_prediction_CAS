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
from keras.layers import LSTM, Dense, Dropout, Input

logging.debug("Imports completed")


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
        self, input_shape, lstm_layers=3, no_units=2, output_units=1, dropout_rate=0.2
    ):
        super(LstmModel, self).__init__()
        self.lstm_layers = []

        self.input_layer = Input(shape=input_shape)
        for i in range(lstm_layers):
            if i == 0:
                self.lstm_layers.append(
                    LSTM(no_units, return_sequences=(lstm_layers > 1))
                )
            else:
                self.lstm_layers.append(
                    LSTM(no_units, return_sequences=(i < lstm_layers - 1))
                )
            self.lstm_layers.append(Dropout(dropout_rate))

        self.output_layer = Dense(output_units)

    def call(self, inputs, training=False):
        x = inputs
        # Apply LSTM and Dropout layers sequentially
        for layer in self.lstm_layers:
            x = layer(x, training=training) if isinstance(layer, Dropout) else layer(x)

        # Final output layer
        return self.output_layer(x)


# Create TensorFlow dataset
def create_dataset(X, y, batch_size=32, shuffle=True):
    """
    X original shape (n,bin,t,band) where:
    n - numebr of samples
    bin - number of bins in each histogram
    t - number of timesteps
    band - spectral bands in the sattelitte photo

    Output:
    dataset shape (n,t,bin*band)
    """
    X = X.reshape(X.shape[0], X.shape[2], X.shape[1] * X.shape[3])
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), X.shape[1:]


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

    # Create MLflow callback to log metrics during training
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

    parser = argparse.ArgumentParser(description="Trains neural network architectures.")
    parser.add_argument(
        "-k", "--key_file", type=str, help="Path to the key file with input parameters."
    )
    args = parser.parse_args()

    if args.key_file:
        params = read_key_file(args.key_file)
        # TODO: Trim down to the parameters needed for LSTM
        dataset_source_dir = params.get("DATASET_FOLDER")
        nnet_architecture = params.get("NNET_ARCHITECTURE")
        season_frac = float(params.get("SEASON_FRAC"))
        permuted_band = float(params.get("PERMUTED_BAND"))
        num_iters = int(params.get("NUM_ITERS"))
        load_weights = params.get("LOAD_WEIGHTS") == "1"
        weights_path_training = (
            params.get("WEIGHTS_PATH_TRAINING") + "/weights/model.weights"
        )
        data_augmentation = params.get("DATA_AUGMENTATION") == "1"
        yield_threshold = float(params.get("YIELD_THRESHOLD"))
        noise_level = float(params.get("NOISE_LEVEL"))
        replication_factor = float(params.get("REPLICATION_FACTOR"))
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
    dev_data = load_data(
        os.path.join(dataset_source_dir, "dev_hists.npz"), dtype=np.float32
    )
    dev_labels = load_data(
        os.path.join(dataset_source_dir, "dev_yields.npz"), dtype=np.float32
    )
    test_data = load_data(
        os.path.join(dataset_source_dir, "test_hists.npz"), dtype=np.float32
    )
    test_labels = load_data(
        os.path.join(dataset_source_dir, "test_yields.npz"), dtype=np.float32
    )

    train_dataset, input_shape = create_dataset(train_data, train_labels)
    val_dataset, _ = create_dataset(test_data, test_labels)

    # Initialize and train the model
    model = LstmModel(input_shape)
    trained_model, history = train_and_evaluate(model, train_dataset, val_dataset)

    # Save the model
    trained_model.save("lstm_model.keras")

    plot_loss(history)

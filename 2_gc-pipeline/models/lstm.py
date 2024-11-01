""" Implemenataion of MLP for crop yield prediction

    Idea:
    Prediction will be made based on the concatenated histograms from all 3 timepoints
"""

import os
from datetime import datetime

import keras
import numpy as np
import randomname
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from utils.plotting import plot_training_progress
from wandb.integration.keras import WandbMetricsLogger


def lstm_data_preparation(data_path: str, labels_path: str):
    """Prepare data for MLP

    Dataset is setup for 3 timepoints for LSTM, ex. shape of single datapoint:
    (1, 3, n_hist_buckets * n_channels)
    """

    dataset = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    return dataset, labels


class Lstm(keras.Sequential):
    def __init__(self, input_shape: tuple, no_lstm_units: list, name: str = None):
        super(Lstm, self).__init__()

        if name is None:
            name = (
                "LSTM-"
                + randomname.get_name(adj=("emotions",), noun=("food"))
                + "-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        super().__init__(name=name)

        self.no_lstm_units = no_lstm_units

        for idx, units in enumerate(no_lstm_units):
            self.add(LSTM(units, return_sequences=idx < len(no_lstm_units) - 1))

        self.add(Dense(1))

        self.model_name = name

    def fit(self, train_dataset, val_dataset, **kwargs):

        train_steps = kwargs.get("steps_per_epoch")
        val_steps = kwargs.get("validation_steps")

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        history = super(Lstm, self).fit(
            train_dataset,
            epochs=kwargs.get("epochs", 100),
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[early_stopping, WandbMetricsLogger()],
        )

        save_dir = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline\models\saved"
        os.makedirs(save_dir, exist_ok=True)
        self.save(f"{save_dir}\\{self.model_name}.keras")

        plot_training_progress(history)

        return history

    def compile(self, **kwargs):
        super().compile(
            optimizer=kwargs.get("optimizer", "adam"),
            loss=kwargs.get("loss", "mse"),
            metrics=kwargs.get("metrics", ["mae"]),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "no_lstm_units": self.no_lstm_units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Extract only the parameters needed for __init__
        init_args = {
            "input_shape": config["input_shape"],
            "no_lstm_units": config["no_lstm_units"],
            "name": config.get("name"),  # Get name if present
        }
        return cls(**init_args)

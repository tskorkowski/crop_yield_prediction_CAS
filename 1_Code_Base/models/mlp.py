""" Implemenataion of MLP for crop yield prediction

    Idea:
    Prediction will be made based on the concatenated histograms from all 3 timepoints
"""

import datetime
import os
from datetime import datetime

import keras
import numpy as np
import randomname
import tensorflow as tf
import wandb
from models.utils import model_save
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from utils.plotting import plot_training_progress
from wandb.integration.keras import WandbMetricsLogger


def mlp_data_preparation(data_path: str, labels_path: str):
    """Prepare data for MLP

    Dataset is setup for 3 timepoints for LSTM, ex. shape of single datapoint:
    (1, 3, n_hist_buckets * n_channels)
    """

    dataset = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    # Concatenate histograms from all 3 timepoints
    dataset = np.concatenate(
        [dataset[:, 0, :], dataset[:, 1, :], dataset[:, 2, :]], axis=-1
    )

    return dataset, labels


class Mlp(keras.Sequential):
    def __init__(
        self,
        input_shape: tuple,
        no_units: list,
        activation: str = "elu",
        name: str = None,
    ):

        if name is None:
            name = (
                "MLP-"
                + randomname.get_name(adj=("emotions",), noun=("food"))
                + "-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        super(Mlp, self).__init__(name=name)

        self.dense_layer_list = no_units
        self.activation = activation
        self.model_name = name

        for units in no_units:
            self.add(Dense(units, activation=activation))
        self.add(Dense(1))

    def fit(self, train_dataset, val_dataset, **kwargs):

        if wandb.run is None:
            wandb.init(
                project="blue-marble",
                config={
                    "architecture": "MLP",
                    "mlp_units": self.dense_layer_list,
                    "learning_rate": kwargs.get("learning_rate", 0.001),
                    "epochs": kwargs.get("epochs", 100),
                },
            )

        train_steps = kwargs.get("steps_per_epoch")
        val_steps = kwargs.get("validation_steps")

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        wandb_callback = WandbMetricsLogger()

        history = super(Mlp, self).fit(
            train_dataset,
            epochs=kwargs.get("epochs", 100),
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[early_stopping, wandb_callback],
        )

        model_save(self)

        plot_training_progress(history)

        wandb.finish()
        return history

    def compile(self, **kwargs):
        super().compile(
            optimizer=kwargs.get("optimizer", "adam"),
            loss=kwargs.get("loss", "mse"),
            metrics=kwargs.get("metrics", ["mae"]),
        )

    def get_config(self):
        return {
            "input_shape": self.input_shape,
            "no_units": self.dense_layer_list,
            "activation": self.activation,
        }

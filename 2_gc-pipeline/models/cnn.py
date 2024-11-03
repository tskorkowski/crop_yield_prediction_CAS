import os
from datetime import datetime

import keras
import numpy as np
import randomname
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from utils.plotting import plot_training_progress
from wandb.integration.keras import WandbMetricsLogger


def cnn_data_preparation(
    dataset_path, labels_path, months: int, num_buckets: int, num_bands: int, **kwargs
):

    dataset = np.load(dataset_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    dataset = dataset.reshape(dataset.shape[0], months, num_buckets, num_bands)
    return dataset, labels


class Cnn2D(keras.Model):
    def __init__(
        self,
        input_shape: tuple,
        conv_filters: list = [32, 64, 128],
        conv_kernel_size: tuple = (1, 3),
        pool_size: tuple = (1, 2),
        dense_units: list = [100, 100, 100],
        dropout_rate: float = 0.2,
        name: str = None,
    ):
        if name is None:
            name = (
                "Cnn2D-"
                + randomname.get_name(adj=("emotions",), noun=("food"))
                + "-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        super().__init__(name=name)

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model_name = name

        self.conv_layers = [
            Conv2D(filters, kernel_size=conv_kernel_size, activation="relu")
            for filters in conv_filters
        ]
        self.pool_layers = [
            MaxPooling2D(pool_size=pool_size) for _ in range(len(conv_filters))
        ]
        self.flatten = Flatten()
        self.dense_layers = [Dense(units, activation="relu") for units in dense_units]
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(1)

    def call(self, inputs):
        x = inputs
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)

        x = self.flatten(x)
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x)

        output = self.output_layer(x)
        return output

    def fit(self, train_dataset, val_dataset, **kwargs):
        if wandb.run is None:
            wandb.init(
                project="blue-marble",
                config={
                    "architecture": "CNN2D",
                    "input_shape": self.input_shape,
                    "conv_filters": self.conv_filters,
                    "conv_kernel_size": self.conv_kernel_size,
                    "pool_size": self.pool_size,
                    "dense_units": self.dense_units,
                    "dropout_rate": self.dropout_rate,
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

        history = super(Cnn2D, self).fit(
            train_dataset,
            epochs=kwargs.get("epochs", 100),
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[early_stopping, wandb_callback],
        )

        save_dir = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline\models\saved"
        os.makedirs(save_dir, exist_ok=True)
        self.save(f"{save_dir}\\{self.model_name}.keras")

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
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "conv_filters": self.conv_filters,
                "conv_kernel_size": self.conv_kernel_size,
                "pool_size": self.pool_size,
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Cnn2DLstm(keras.Model):
    def __init__(
        self,
        input_shape: tuple,
        conv_filters: list = [32, 64, 128],
        conv_kernel_size: tuple = (1, 3),
        pool_size: tuple = (1, 2),
        dense_units: list = [100, 100, 100],
        dropout_rate: float = 0.2,
        name: str = None,
    ):
        if name is None:
            name = (
                "CNN2D-"
                + randomname.get_name(adj=("emotions",), noun=("food"))
                + "-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        super().__init__(name=name)

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model_name = name

        self.conv_layers = [
            keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    filters, kernel_size=conv_kernel_size, activation="relu"
                )
            )
            for filters in conv_filters
        ]
        self.pool_layers = [
            keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=pool_size))
            for _ in range(len(conv_filters))
        ]
        self.flatten = keras.layers.TimeDistributed(keras.layers.Flatten())
        self.dense_layers = [Dense(units, activation="relu") for units in dense_units]
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(1)

    def call(self, inputs):
        x = inputs
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)

        x = self.flatten(x)
        x = keras.layers.LSTM(64)(
            x
        )  # Add an LSTM layer to capture temporal dependencies
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x)

        output = self.output_layer(x)
        return output

    def fit(self, train_dataset, val_dataset, **kwargs):
        if wandb.run is None:
            wandb.init(
                project="blue-marble",
                config={
                    "architecture": "CNN2DLSTM",
                    "input_shape": self.input_shape,
                    "conv_filters": self.conv_filters,
                    "conv_kernel_size": self.conv_kernel_size,
                    "pool_size": self.pool_size,
                    "dense_units": self.dense_units,
                    "dropout_rate": self.dropout_rate,
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

        history = super(Cnn2DLstm, self).fit(
            train_dataset,
            epochs=kwargs.get("epochs", 100),
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=[early_stopping, wandb_callback],
        )

        save_dir = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline\models\saved"
        os.makedirs(save_dir, exist_ok=True)
        self.save(f"{save_dir}\\{self.model_name}.keras")

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
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "conv_filters": self.conv_filters,
                "conv_kernel_size": self.conv_kernel_size,
                "pool_size": self.pool_size,
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

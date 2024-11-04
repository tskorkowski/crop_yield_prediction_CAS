import os
from datetime import datetime

import keras
import numpy as np
import randomname
import tensorflow as tf
import wandb
from models.lstm import AttentionLayer
from models.utils import model_save
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    ConvLSTM1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
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
        l1_reg: float = 0.01,
        l2_reg: float = 0.01,
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
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        regularizer = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)

        self.conv_layers = [
            Conv2D(
                filters,
                kernel_size=conv_kernel_size,
                activation="relu",
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                data_format="channels_first",
            )
            for filters in conv_filters
        ]
        self.pool_layers = [
            MaxPooling2D(pool_size=pool_size) for _ in range(len(conv_filters))
        ]
        self.flatten = Flatten()
        self.dense_layers = [
            Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
            )
            for units in dense_units
        ]
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
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "conv_filters": self.conv_filters,
                "conv_kernel_size": self.conv_kernel_size,
                "pool_size": self.pool_size,
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
                "l1_reg": self.l1_reg,
                "l2_reg": self.l2_reg,
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
        no_lstm_units: list = [100, 100],
        attention_units: int = 100,
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
        self.attention_units = attention_units

        # Create individual layers for each time step
        self.conv_blocks = []
        for _ in range(len(conv_filters)):
            conv_block = keras.Sequential(
                [
                    Conv2D(
                        filters=conv_filters[_],
                        kernel_size=conv_kernel_size,
                        activation="relu",
                        padding="same",
                        # input_shape=(
                        #     input_shape[1],
                        #     input_shape[2],
                        #     1,
                        # ),  # Specify input shape
                    ),
                    MaxPooling2D(pool_size=pool_size, padding="same"),
                ]
            )
            self.conv_blocks.append(keras.layers.TimeDistributed(conv_block))

        self.flatten = keras.layers.TimeDistributed(Flatten())
        self.dense = [Dense(units, activation="relu") for units in no_lstm_units]
        # self.dense_layers = [Dense(units, activation="relu") for units in dense_units]
        self.attention = AttentionLayer(attention_units)
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(1)

    def call(self, inputs):
        # Add channel dimension
        x = tf.expand_dims(inputs, axis=-1)

        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)

        for lstm in self.lstm_layers:
            x = lstm(x)

        # for dense in self.dense_layers:
        #     x = dense(x)
        #     x = self.dropout(x)
        # Apply attention
        query = x[:, -1, :]  # Use the last hidden state as the query
        context_vector, attention_weights = self.attention(query, x)

        # Concatenate the context vector with the last hidden state
        x = tf.concat([context_vector, x[:, -1, :]], axis=-1)

        return self.output_layer(x)

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

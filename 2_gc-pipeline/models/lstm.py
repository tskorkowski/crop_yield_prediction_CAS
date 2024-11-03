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
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    GlobalAveragePooling1D,
    MultiHeadAttention,
)
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

        if wandb.run is None:
            wandb.init(
                project="blue-marble",
                config={
                    "architecture": "LSTM",
                    "input_shape": self.input_shape,
                    "lstm_units": self.no_lstm_units,
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

        history = super(Lstm, self).fit(
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


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class LstmWithAttention(keras.Model):
    def __init__(
        self,
        input_shape: tuple,
        no_lstm_units: list,
        attention_units: int,
        name: str = None,
        **kwargs,
    ):
        super(LstmWithAttention, self).__init__(name=name)

        self.input_shape = input_shape
        self.no_lstm_units = no_lstm_units
        self.attention_units = attention_units
        self.model_name = name

        self.lstm_layers = [
            LSTM(units, return_sequences=True) for units in no_lstm_units
        ]
        self.attention = AttentionLayer(attention_units)
        self.dense = Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)

        # Apply attention
        query = x[:, -1, :]  # Use the last hidden state as the query
        context_vector, attention_weights = self.attention(query, x)

        # Concatenate the context vector with the last hidden state
        x = tf.concat([context_vector, x[:, -1, :]], axis=-1)

        # Pass through the final Dense layer
        output = self.dense(x)

        return output

    def fit(self, train_dataset, val_dataset, **kwargs):

        if wandb.run is None:
            wandb.init(
                project="blue-marble",
                config={
                    "architecture": "LSTM",
                    "input_shape": self.input_shape,
                    "lstm_units": self.no_lstm_units,
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

        history = super(LstmWithAttention, self).fit(
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
                "no_lstm_units": self.no_lstm_units,
                "attention_units": self.attention_units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

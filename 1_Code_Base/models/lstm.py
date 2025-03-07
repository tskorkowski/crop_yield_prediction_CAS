"""Implemenataion of MLP for crop yield prediction

Idea:
Prediction will be made based on the concatenated histograms from all 3 timepoints
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd
import randomname
import tensorflow as tf
import wandb
from models.utils import model_save
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Normalization,
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

    def build(self, input_shape):  # Add proper build method
        super().build(input_shape)

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
                "no_lstm_units": self.no_lstm_units,
                "attention_units": self.attention_units,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PreprocessingHead(tf.keras.Model):
    """Preprocessing model for weather and satellite data
    * Normalizes the data
    * Encoding of categorical features
    """

    def __init__(self, input_dict: Dict[str, np.ndarray], cat_features: List[str]):
        super().__init__()

        self.normalizer = Normalization(axis=-1)
        self.concat_num_features = Concatenate()
        self.concat_all_features = Concatenate()

        self._cat_features = cat_features
        self.inputs = {}

        self._numeric_features = []
        self._numeric_features_dict = {}

        for name, values in input_dict.items():
            if name in self._cat_features:
                dtype = tf.string
            else:
                dtype = tf.float64
                self._numeric_features.append(name)
                self._numeric_features_dict[name] = values

            self.inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        self.normalizer.adapt(
            np.concatenate(
                [
                    np.expand_dims(value, axis=1)
                    for _, value in sorted(self._numeric_features_dict.items())
                ],
                axis=1,
            )
        )

        self.cat_encoders = {}
        for feature in self._cat_features:

            unique_values = np.unique(input_dict[feature])
            one_hot_encoder = tf.keras.layers.StringLookup(
                vocabulary=unique_values, output_mode="one_hot"
            )
            self.cat_encoders[feature] = one_hot_encoder

    def call(self, inputs):
        """_summary_

        Args:
            inputs (dict): should be a dict of numpy arrays representing the input df

        Returns:
            _type_: _description_
        """
        # Process numeric features
        numeric_features = []
        for feature in sorted(self._numeric_features):
            numeric_features.append(tf.expand_dims(inputs[feature], axis=1))

        numeric_tensor = self.concat_num_features(numeric_features)

        normalized_features = self.normalizer(numeric_tensor)

        # Process categorical features
        categorical_features = []
        for feature in self._cat_features:
            encoded = self.cat_encoders[feature](inputs[feature])
            categorical_features.append(encoded)

        # If there are categorical features, concatenate them with normalized features
        if categorical_features:
            all_features = categorical_features + [normalized_features]
            return self.concat_all_features(all_features)
        else:
            return normalized_features


class Embeddings(tf.keras.Model):
    """
    Embeddings layer for weather and satellite data
    """

    def __init__(self, dense_layer_list: List[int]):
        super(Embeddings, self).__init__()
        self.dense_layers = [Dense(layer) for layer in dense_layer_list]

        self.dropout_layers = [Dropout(0.2) for _ in dense_layer_list]

    def call(self, inputs):
        x = self.dense_layers[0](inputs)
        x = self.dropout_layers[0](x)
        for layer, dropout in zip(self.dense_layers[1:], self.dropout_layers[1:]):
            x = layer(x)
            x = dropout(x)
        return x


class LstmWeather(tf.keras.Model):
    """
    LSTM model for weather and satellite data

    Creates separate embeddings for weather and satellite data. Both embeddings are concatenated and passed to the LSTM.
    """

    def __init__(
        self,
        weather_datasets: List[Dict[str, np.ndarray]],
        sat_datasets: List[Dict[str, np.ndarray]],
        cat_features: List[str],
        lstm_units: int = 1,
        embedings_spec_weather: List[int] = [256, 128, 64, 32, 16],
        embedings_spec_satellite: List[int] = [256, 128, 64, 32, 16],
        timepoints: int = 3,
    ):

        super().__init__()

        self.prepocessing_weather = [
            PreprocessingHead(weather_dataset, cat_features)
            for weather_dataset in weather_datasets
        ]
        self.prepocessing_satellite = [
            PreprocessingHead(sat_dataset, cat_features) for sat_dataset in sat_datasets
        ]

        self.weather_embeddings = [
            Embeddings(embedings_spec_weather) for _ in range(timepoints)
        ]
        self.satellite_embeddings = [
            Embeddings(embedings_spec_satellite) for _ in range(timepoints)
        ]

        self.concatenate_weather_and_sat_embeddings = [
            Concatenate(axis=1) for _ in range(timepoints)
        ]

        self.lstm = LSTM(lstm_units, return_sequences=False)

        self.dense = Dense(1, activation="exponential")

        self.timepoints = timepoints

    def call(self, inputs):
        weather_data, satellite_data = inputs

        processed_weather_data = [
            self.prepocessing_weather[i](weather_data[i])
            for i in range(self.timepoints)
        ]
        processed_satellite_data = [
            self.prepocessing_satellite[i](satellite_data[i])
            for i in range(self.timepoints)
        ]

        weather_embeddings = [
            embedding(data)
            for embedding, data in zip(self.weather_embeddings, processed_weather_data)
        ]

        satellite_embeddings = [
            embedding(data)
            for embedding, data in zip(
                self.satellite_embeddings, processed_satellite_data
            )
        ]
        weather_and_satellite_embeddings = [
            weather_and_sat_embedding([weather_embeddings[i], satellite_embeddings[i]])
            for i, weather_and_sat_embedding in enumerate(
                self.concatenate_weather_and_sat_embeddings
            )
        ]

        lstm_input = tf.stack(weather_and_satellite_embeddings, axis=0)
        lstm_input = tf.transpose(lstm_input, perm=[1, 0, 2])

        lstm_output = self.lstm(lstm_input)

        output = self.dense(lstm_output)

        return output


# TODO:
# https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M

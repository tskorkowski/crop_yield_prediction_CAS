""" Code modified from https://github.com/JiaxuanYou/crop_yield_prediction"""

import math
import time
from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LSTM_Config():
    B, W, C = 32,32,9
    season_len = 39
    H = season_len # Number of steps in the input sequence
    loss_lambda = 0.75
    lstm_layers = 1
    lstm_H = 200
    dense = 356

    train_step = 10000
    lr = 0.0005
    #keep probability
    drop_out = 0.75

    def __init__(self, season_frac=None):
        if season_frac is not None:
            self.H = int(season_frac*self.H)
    
def create_lstm_model(config):
    model = keras.Sequential([
        keras.layers.LSTM(config.lstm_H, return_sequences=True, 
                          input_shape=(config.H, config.W * config.C)),
        keras.layers.LSTM(config.lstm_H),
        keras.layers.Dropout(1 - config.drop_out),
        keras.layers.Dense(config.dense, activation='relu'),
        keras.layers.Dense(1)
    ])
    return model

class LSTM_NeuralModel():
    def __init__(self, config, name):
        self.model = create_lstm_model(config)
        self.optimizer = keras.optimizers.Adam(learning_rate=config.lr)
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value

    def predict(self, x):
        return self.model(x, training=False)

    def add_finetuning_layer(self, config_dense, config_loss_lambda, config_lr):
        # Freeze the existing layers
        for layer in self.model.layers:
            layer.trainable = False
        
        # Add new layers for fine-tuning
        self.model.add(keras.layers.Dense(config_dense, activation='relu'))
        self.model.add(keras.layers.Dense(1))
        
        # Update optimizer and compile the model
        self.optimizer = keras.optimizers.Adam(learning_rate=config_lr)
        self.model.compile(optimizer=self.optimizer, loss='mse')
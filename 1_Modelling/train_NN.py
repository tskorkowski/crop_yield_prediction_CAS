import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from constants import DATASETS, GBUCKET
from nnet_CNN import CNN_Config, CNN_NeuralModel
# Import your model configurations and definitions
from nnet_LSTM import LSTM_Config, LSTM_NeuralModel
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tensorflow import keras
from util import Progbar

t = time.localtime()
timeString = time.strftime("%Y-%m-%d_%H-%M-%S", t)

def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()
    return params

def augment_data(inputs, labels, low_yield_threshold=1.8, noise_level=0.01, replication_factor=5):
    inputs = inputs.astype(np.float32)
    labels = labels.astype(np.float32)

    low_yield_indices = np.where(labels < low_yield_threshold)[0]
    augmented_inputs = np.copy(inputs)
    augmented_labels = np.copy(labels)

    for index in low_yield_indices:
        for _ in range(int(replication_factor)):
            noisy_input = inputs[index] + noise_level * np.random.randn(*inputs[index].shape)
            augmented_inputs = np.append(augmented_inputs, noisy_input[np.newaxis, ...], axis=0)
            augmented_labels = np.append(augmented_labels, labels[index])

    return augmented_inputs, augmented_labels

def load_data(file_path):
    try:
        data = np.load(file_path)['data']
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_dataset(inputs, labels, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def run_NN(model, directory, CNN_or_LSTM, config, output_google_doc, restrict_iterations=None, rows_to_add_to_google_doc=None, permuted_band=-1):
    train_name = f"train_{CNN_or_LSTM}_{config.lr}_{config.drop_out}_{config.train_step}_{timeString}"
    train_dir = os.path.expanduser(os.path.join('Training', train_name))
    os.makedirs(train_dir, exist_ok=True)

    train_logfile = os.path.join(train_dir, f'{train_name}.log')
    logging.basicConfig(filename=train_logfile, level=logging.DEBUG)

    # Load data
    train_data = load_data(os.path.join(directory, 'train_hists.npz'))
    train_labels = load_data(os.path.join(directory, 'train_yields.npz'))
    dev_data = load_data(os.path.join(directory, 'dev_hists.npz'))
    dev_labels = load_data(os.path.join(directory, 'dev_yields.npz'))
    test_data = load_data(os.path.join(directory, 'test_hists.npz'))
    test_labels = load_data(os.path.join(directory, 'test_yields.npz'))

    if data_augmentation:
        train_data, train_labels = augment_data(train_data, train_labels, 
                                                low_yield_threshold=yield_threshold, 
                                                noise_level=noise_level, 
                                                replication_factor=replication_factor)

    train_dataset = create_dataset(train_data, train_labels, config.B)
    
    # Metrics
    train_loss_metric = keras.metrics.Mean()
    train_rmse_metric = keras.metrics.RootMeanSquaredError()

    best_dev_rmse = float('inf')
    best_test_rmse = float('inf')

    num_iters = restrict_iterations or 10000
    progbar = Progbar(target=num_iters)

    for epoch in range(num_iters):
        for x_batch, y_batch in train_dataset:
            loss = model.train_step(x_batch, y_batch)
            train_loss_metric.update_state(loss)
            train_rmse_metric.update_state(y_batch, model.predict(x_batch))

        if epoch % 100 == 0:
            train_loss = train_loss_metric.result().numpy()
            train_rmse = train_rmse_metric.result().numpy()
            train_pred = model.predict(train_data)
            train_r2 = r2_score(train_labels, train_pred)

            logging.info(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}')

            if dev_data is not None:
                dev_pred = model.predict(dev_data)
                dev_rmse = np.sqrt(np.mean((dev_pred - dev_labels) ** 2))
                dev_r2 = r2_score(dev_labels, dev_pred)
                logging.info(f'Dev RMSE: {dev_rmse:.4f}, R2: {dev_r2:.4f}')

                if dev_rmse < best_dev_rmse:
                    best_dev_rmse = dev_rmse
                    model.model.save(os.path.join(train_dir, 'best_model_dev'))

            if test_data is not None:
                test_pred = model.predict(test_data)
                test_rmse = np.sqrt(np.mean((test_pred - test_labels) ** 2))
                test_r2 = r2_score(test_labels, test_pred)
                logging.info(f'Test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}')

                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    model.model.save(os.path.join(train_dir, 'best_model_test'))

            train_loss_metric.reset_states()
            train_rmse_metric.reset_states()

        progbar.update(epoch + 1, [("train loss", train_loss)])

    # Final save
    model.model.save(os.path.join(train_dir, 'final_model'))

    return best_dev_rmse, best_test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains neural network architectures.')
    parser.add_argument("-k", "--key_file", type=str, help="Path to the key file with input parameters.")
    args = parser.parse_args()

    if args.key_file:
        params = read_key_file(args.key_file)
        dataset_source_dir = params.get('DATASET_FOLDER')
        nnet_architecture = params.get('NNET_ARCHITECTURE')
        season_frac = float(params.get('SEASON_FRAC'))
        permuted_band = float(params.get('PERMUTED_BAND'))
        num_iters = int(params.get('NUM_ITERS'))
        load_weights = params.get('LOAD_WEIGHTS') == '1'
        weights_path_training = params.get('WEIGHTS_PATH_TRAINING') + "/weights/model.weights"
        data_augmentation = params.get('DATA_AUGMENTATION') == '1'
        yield_threshold = float(params.get('YIELD_THRESHOLD'))
        noise_level = float(params.get('NOISE_LEVEL'))
        replication_factor = float(params.get('REPLICATION_FACTOR'))
    else:
        print("Key file is required.")
        sys.exit(1)

    if nnet_architecture == 'CNN':
        config = CNN_Config(season_frac)
        model = CNN_NeuralModel(config, 'net')
        print("Running CNN...")
    elif nnet_architecture == 'LSTM':
        config = LSTM_Config(season_frac)
        model = LSTM_NeuralModel(config, 'net')
        print("Running LSTM...")
    else:
        print('Error: did not specify a valid neural network architecture. Ending..')
        sys.exit()

    if load_weights:
        model.model.load_weights(weights_path_training)
        print(f"Loaded weights from {weights_path_training}")

    experiment_doc_name = ""  # Add your experiment doc name if needed
    best_dev_rmse, best_test_rmse = run_NN(model, dataset_source_dir, nnet_architecture, config, 
                                           experiment_doc_name, restrict_iterations=num_iters, 
                                           permuted_band=permuted_band)

    print(f"Training completed. Best Dev RMSE: {best_dev_rmse:.4f}, Best Test RMSE: {best_test_rmse:.4f}")
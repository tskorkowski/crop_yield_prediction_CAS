# To run: python3 yield_prediction.py -k keyfile.key

import tensorflow as tf
import argparse
import numpy as np
import os
import sys
import csv
from nnet_LSTM import LSTM_NeuralModel, LSTM_Config
from nnet_CNN import CNN_NeuralModel, CNN_Config

def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):  # Ignore comments and empty lines
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()  # This also removes comments from the value, if any
    return params

def file_generator(inputs_data_path, labels_data_path, batch_size, add_data_path = None, add_labels_path = None, permuted_band = -1):
    current_batch_labels = []
    current_batch_inputs = []
    inputs_data = np.load(inputs_data_path)['data']
    labels_data = np.load(labels_data_path)['data']

    if permuted_band != -1:
        n_training_examples = inputs_data.shape[0]
        examples_permutation = np.random.permutation(n_training_examples)
        inputs_data[:, :, :, permuted_band] = inputs_data[examples_permutation, :, :, permuted_band]

    assert(len(inputs_data) == len(labels_data))
    if add_data_path is not None:
        inputs_data_2 = np.load(add_data_path)['data']
        inputs_data = np.append(inputs_data, inputs_data_2, axis = 0)
    if add_labels_path is not None:
        labels_data_2 = np.load(add_labels_path)['data']
        labels_data = np.append(labels_data, labels_data_2, axis = 0)
    
    while(len(inputs_data) < batch_size):
        #need to pad
        inputs_data = np.append(inputs_data, inputs_data, axis = 0)
        labels_data = np.append(labels_data, labels_data, axis = 0)
        print("appended to pad")
    for idx in range(len(inputs_data)):
       current_batch_inputs.append(inputs_data[idx])
       current_batch_labels.append(labels_data[idx]) 
       if len(current_batch_labels) == batch_size:
            yield current_batch_inputs, current_batch_labels
            current_batch_inputs = []
            current_batch_labels = []


def load_model(sess, model_path, model):
    model.saver.restore(sess, model_path)
    print(f"Model weights loaded from {model_path}")

def make_predictions(sess, model, inputs_data_path, modis_labels_file, config, permuted_band):
    all_predictions = []
    for batch in file_generator(inputs_data_path, modis_labels_file, config.B, permuted_band=permuted_band):
        pred_temp = sess.run(model.pred, feed_dict={
            model.x: batch[0],
            model.y: batch[1],
            model.keep_prob: 1
        })
        all_predictions.extend(pred_temp)  # Accumulate predictions
    return all_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make yield predictions with trained CNN or LSTM model.')
    parser.add_argument("-k", "--key_file", type=str, help="Path to the key file with input parameters.")

    args = parser.parse_args()

    if args.key_file:
            params = read_key_file(args.key_file)
            model_type = params.get('NNET_ARCHITECTURE')
            weights_path = params.get('WEIGHTS_PATH')+"/weights/model.weights"
            modis_data_file = params.get('DATASET_FOLDER')+"/hists.npz"
            modis_labels_file = params.get('DATASET_FOLDER')+"/yields.npz"
            season_frac = float(params.get('SEASON_FRAC'))
            permuted_band = float(params.get('PERMUTED_BAND'))

    else:
            print("Key file is required.")
            sys.exit(1)

    years = np.load(params.get('DATASET_FOLDER')+"/years.npz")
    years = years[years.files[0]]
    
    if model_type == 'CNN':
        config = CNN_Config(season_frac)
        model= CNN_NeuralModel(config,'net')
        print("Running CNN...")
    elif model_type == 'LSTM':
        config = LSTM_Config(season_frac)
        model= LSTM_NeuralModel(config,'net')
        print("Running LSTM...")
    else:
        print('Error: did not specify a valid neural network architecture. Ending..')
        sys.exit()
    
    model.summary_op = tf.compat.v1.summary.merge_all()
    model.saver = tf.compat.v1.train.Saver()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.22)

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.compat.v1.initialize_all_variables())
    model.saver.restore(sess, weights_path)

    predictions = make_predictions(sess, model, modis_data_file, modis_labels_file, config, permuted_band)
    real_yields = np.load(modis_labels_file)['data']
    seen = set()
    predictions = np.array([x for x in predictions if x not in seen and not seen.add(x)])
    paired = list(zip(years, predictions,real_yields))
    paired_sorted = sorted(paired)
    years_sorted, predictions_sorted, real_yields_sorted = zip(*paired_sorted)
    years_sorted, predictions_sorted, real_yields_sorted = list(years_sorted), list(predictions_sorted), list(real_yields_sorted)
    


    rel_error =[]
    for i in range(len(predictions_sorted)):
        print(years_sorted[i],"-->", predictions_sorted[i], ' Real value:', real_yields_sorted[i] ,'   Relative error: ', np.round(100*np.abs(real_yields_sorted[i] - predictions_sorted[i]) / ((real_yields_sorted[i] + predictions_sorted[i]) / 2),0 ),'%' )
        rel_error.append(np.round(100*np.abs(real_yields_sorted[i] - predictions_sorted[i]) / ((real_yields_sorted[i] + predictions_sorted[i]) / 2),0 ))
    print('Mean relative error is ', np.mean(rel_error),'%.')

    # Save predictions_sorted and real_yields_sorted to CSV
    csv_file = "Analysis/predictions.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prediction", "Real Yield"])
        for pred, real in zip(predictions_sorted, real_yields_sorted):
            writer.writerow([pred, real])
    
    print(f"Predictions and real yields saved to {csv_file}.")


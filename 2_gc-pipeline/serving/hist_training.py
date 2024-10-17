import glob
import os
import io
import numpy as np
import pandas as pd
from serving.constants import SCALE, NUM_BINS, NUM_BANDS, HIST_DEST_PREFIX, BUCKET # meters per pixel, number of bins in the histogram, number of bands in the satellite image
import tensorflow as tf
import itertools
from google.cloud import storage
import google.auth
import logging
import logging
import os
import random
import sys

import keras
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Input


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

def combine_crop_data(path, save=False):

    # header:
    header = [
        "commodity_desc",
        "reference_period_desc",
        "year",
        "state_name",
        "county_name",
        "YIELD, MEASURED IN BU / ACRE",
    ]

    # Use glob to get all the csv files in the directory and its subdirectories
    all_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

    # Read each file into a dataframe
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, usecols=header, index_col=None, header=0)
        df["source_file"] = os.path.basename(filename)
        df_list.append(df)

    # Combine all dataframes
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    combined_df.rename(columns={"YIELD, MEASURED IN BU / ACRE": "target"}, inplace=True)

    # Perform any necessary data cleaning or transformation here
    # For example:
    # combined_df['date'] = pd.to_datetime(combined_df['date'])
    # combined_df = combined_df.dropna()

    # Convert to numpy arrays
    combined_labels = combined_df.to_numpy()

    if save:
        # Save as numpy arrays
        np.save("labels_combined.npy", combined_labels)

        # Optionally, save column names for reference
        np.save("labels_header.npy", combined_df.columns.to_numpy())

    return combined_df


def create_hist_dataset(counties: list, years: list, months: list, labels_path: str, header_path: str) ->  tf.data.Dataset:
    
    logging.basicConfig(filename=f"{__name__}" ,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    client = storage.Client()
    bucket = client.get_bucket(BUCKET) 
    
    hist_name_base = f"{HIST_DEST_PREFIX}"
    histograms = []
    labels = []
    combinations = list(itertools.product(counties, years, months))
    filtered_combinations = []
    processed_county_year = set()
    
    label_data = np.load(labels_path, allow_pickle=True)
    label_header = np.load(header_path, allow_pickle=True)
    label_df = pd.DataFrame(label_data, columns=label_header)    
    data_counties = set()
    data_years = set()
    
    for combination in combinations:
        county, year, month = combination
        try:
            label = (label_df.loc[(label_df["county_name"] == county.upper()) & (label_df["year"] == year), "target"].iloc[0])        
            filtered_combinations.append(combination)
            
            if f"{county}/{year}" in processed_county_year:
                continue
            else:
                processed_county_year.add(f"{county}/{year}")
                labels.append(label)
            
        except IndexError:
            logging.info(f"""County {county.upper()} in {year} does not exist in ground truth data. Choose different time/region.
            This histogram will be ignored""")
            continue
                
    for combination in filtered_combinations:
        county, year, month = combination
        file_name = f"{hist_name_base}/{county}/{year}/{month}-{month+1}/{SCALE}/hist.npy"
        hist_blob = bucket.blob(file_name) 
    
        # Training points
        if hist_blob.name.endswith('.npy'):
            # Download the content of the blob
            content = hist_blob.download_as_bytes()

            # Use BytesIO to create a file-like object in memory
            binary_data = io.BytesIO(content)

            # Load the NumPy array from the binary data
            array = np.load(binary_data)

        else:
            array = np.zeros(NUM_BINS * NUM_BANDS) 
        
        histograms.append(array)
        

            
    # Create tensors
    
    histogram_tensor = tf.convert_to_tensor(histograms, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
    
    # Reshape the tensor to (num_examples, len(months), NUM_BINS * NUM_BANDS)
    reshaped_tensor = tf.reshape(histogram_tensor, (-1, len(months), NUM_BINS * NUM_BANDS))
    print(reshaped_tensor.shape, label_tensor.shape, len(filtered_combinations), sep="\n")
    assert reshaped_tensor.shape[0] == len(processed_county_year), "somethign went wrong when aggreagting training data"
    
    # Create a TensorFlow dataset from the reshaped tensor
    dataset = tf.data.Dataset.from_tensor_slices((reshaped_tensor, label_tensor))
    
    return dataset, reshaped_tensor.shape[1:]


if __name__ == "__main__":
    path = "2_gc-pipeline/serving_hist/USDA/data/Corn"
    df = combine_crop_data(path, save=False)
    print(df.head())

    train_dataset, input_shape = create_dataset(train_data, train_labels)
    val_dataset, _ = create_dataset(test_data, test_labels)

    # Initialize and train the model
    model = LstmModel(input_shape)
    trained_model, history = train_and_evaluate(model, train_dataset, val_dataset)

    # Save the model
    trained_model.save("lstm_model.keras")

    plot_loss(history)
    
import glob
import os
import io
import numpy as np
import pandas as pd
from serving.constants import SCALE, NUM_BINS, NUM_BANDS, HIST_DEST_PREFIX, BUCKET, LABELS_PATH, HEADER_PATH # meters per pixel, number of bins in the histogram, number of bands in the satellite image
from serving.data import get_labels
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
        self, input_shape, lstm_layers=3, no_units=3, output_units=1, dropout_rate=0.2
    ):
        super(LstmModel, self).__init__()
        
        self.normalizer = tf.keras.layers.Normalization(axis=-1)  # Add the normalizer to your model        
        self.lstm_layers = []
        

        self.input_layer = Input(shape=(None, input_shape[0], input_shape[1]))
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
        x = self.normalizer(inputs)
        # Apply LSTM and Dropout layers sequentially
        for layer in self.lstm_layers:
            x = layer(x, training=training) if isinstance(layer, Dropout) else layer(x)

        # Final output layer
        return self.output_layer(x)


# Main training loop
def train_and_evaluate(
    model, dataset, validation_split=0.1, epochs=10, learning_rate=0.001, patience=10
):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=True)

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
            
    
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.8 * dataset_size)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)            

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, MLflowCallback()],
        verbose=1
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

def create_hist_dataset(counties: list, years: list, months: list, labels_path: str=LABELS_PATH, header_path: str=HEADER_PATH) -> tf.data.Dataset:
    
    logging.basicConfig(filename=f"{__name__}" ,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    client = storage.Client()
    bucket = client.get_bucket(BUCKET) 
    
    hist_name_base = f"{HIST_DEST_PREFIX}"
    histograms = []
    labels = []
    combinations = list(itertools.product(counties, years, months))
    filtered_combinations = []
    processed_county_year = set()
    
    label_df = get_labels(labels_path, header_path)
    
    for combination in combinations:
        county, year, month = combination
        try:
            label = float(label_df.loc[(label_df["county_name"] == county.upper()) & (label_df["year"] == year), "target"].iloc[0])
            filtered_combinations.append(combination)
            
            if f"{county}/{year}" not in processed_county_year:
                processed_county_year.add(f"{county}/{year}")
                labels.append(label)
            
        except IndexError:
            logging.info(f"County {county.upper()} in {year} does not exist in ground truth data. This histogram will be ignored")
            continue
                
    for combination in filtered_combinations:
        county, year, month = combination
        file_name = f"{hist_name_base}/{county}/{year}/{month}-{month+1}/{SCALE}/hist.npy"
        hist_blob = bucket.blob(file_name) 
    
        if hist_blob.name.endswith('.npy'):
            content = hist_blob.download_as_bytes()
            binary_data = io.BytesIO(content)
            array = np.load(binary_data)
        else:
            array = np.zeros(NUM_BINS * NUM_BANDS) 
        
        histograms.append(array)
    
    # Convert lists to numpy arrays
    histograms_np = np.array(histograms, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)
    
    # Reshape the histograms array
    reshaped_histograms = histograms_np.reshape(-1, len(months), NUM_BINS * NUM_BANDS)
    
    # Ensure labels are 2D
    labels_np = labels_np.reshape(-1, 1)
    
    print(f"Reshaped histograms shape: {reshaped_histograms.shape}")
    print(f"Labels shape: {labels_np.shape}")
    print(f"Number of filtered combinations: {len(filtered_combinations)}")
    
    assert reshaped_histograms.shape[0] == len(processed_county_year), "Something went wrong when aggregating training data"
    
    # Create TensorFlow tensors
    histogram_tensor = tf.convert_to_tensor(reshaped_histograms, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(labels_np, dtype=tf.float32)
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((histogram_tensor, label_tensor))
    
    return dataset, reshaped_histograms.shape[1:]

def create_data_sample():
    # Example: X is a list of NumPy arrays, y is a list of corresponding labels
    X = [np.random.rand(3, 416), np.random.rand(3, 416), np.random.rand(3, 416)]  # Input features (e.g., 3 timesteps, 416 features)
    y = [165.5, 179.7, 172.5]  # Corresponding labels

    # Step 1: Convert the lists to NumPy arrays (optional but recommended for consistency)
    X = np.array(X)
    y = np.array(y)

    # Step 2: Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # Step 3: Shuffle and batch the dataset
    batch_size = 2  # Set the batch size
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)

    # Step 4: Inspect the dataset
    for data_point, label in dataset.take(1):
        print("Data point shape (after batching):", data_point.shape)
        print("Label shape (after batching):", label.shape)
    
    return dataset

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
    
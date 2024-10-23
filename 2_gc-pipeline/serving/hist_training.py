import glob
import os
import io
import numpy as np
import pandas as pd
from serving.constants import SCALE, NUM_BINS, NUM_BANDS, HIST_DEST_PREFIX, BUCKET, LABELS_PATH, HEADER_PATH, MONTHS # meters per pixel, number of bins in the histogram, number of bands in the satellite image
from serving.data import get_labels
import tensorflow as tf
import itertools
from google.cloud import storage
import google.auth
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
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt

def set_seeds(seed=42):
    # Set seeds for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for TensorFlow
    tf.random.set_seed(seed)

  
# Define the LSTM model
class LstmModel(keras.Model):
    def __init__(self, input_shape, lstm_layers=3, no_units=3, output_units=1, dropout_rate=0.2, mean_response=0):
        super(LstmModel, self).__init__()

        self.inputs = Input(shape=input_shape)
        self.lstm_layers = lstm_layers

        self.model = self.build_model(no_units)
        
    def build_model(self, no_lstm_units):
        
        #LSTM layers
        x = self.inputs
        for i in range(self.lstm_layers):
            return_sequences = (i < self.lstm_layers - 1)
            x = LSTM(units=no_lstm_units, return_sequences=return_sequences, kernel_initializer='zeros', recurrent_initializer='zeros', bias_initializer='zeros')(x)
        
        outputs = Dense(units=output_units, kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(mean_response))(x)
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)        
        return model

    def summary(self):
        self.model.summary()
    
    def compile(self, optimizer='adam', loss='mse', metrics=['accuracy'], learning_rate=0.001):
        self.learning_rate = learning_rate
        if optimizer == 'nadam':
            optimizer = Nadam(learning_rate)
        elif optimizer == 'rms':
            optimizer = RMSprop(learning_rate)
        else:
            optimizer = Adam(learning_rate)
            
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def fit(self, dataset, epochs=10, batch_size=32):
        # Set up MLflow experiment
        mlflow.set_experiment('LSTM_Experiment')
        
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)

        # Split the dataset
        val_dataset = dataset.take(5)
        train_dataset = dataset.skip(val_size)        

        # Start MLflow run
        with mlflow.start_run():
            # Early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Create a history callback
            history_callback = tf.keras.callbacks.History()
           
            # Train the model
            history = self.model.fit(train_dataset, epochs=epochs, batch_size=batch_size,
                           validation_data=val_dataset, callbacks=[early_stopping, history_callback])

            # Log model and parameters to MLflow
            mlflow.keras.log_model(self.model, 'model')
            mlflow.log_param('epochs', epochs)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('lstm_layers', self.lstm_layers)
            mlflow.log_param('learning_rate', self.learning_rate)
            
            # Plot training progress
            self.plot_training_progress(history)            

            # Evaluate the model
            loss, accuracy = self.model.evaluate(val_dataset)
            mlflow.log_metric('val_loss', loss)
            mlflow.log_metric('val_accuracy', accuracy)

    def evaluate(self, dataset_test):
        loss, accuracy = self.model.evaluate(dataset_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

    def plot_training_progress(self, history):
        # Plot training and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        plt.show()        

    @tf.function
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


# Main training loop
def train_and_evaluate(
    model, dataset, validation_split=0.2, epochs=10, initial_learning_rate=0.001, patience=10, batch_size=32
):

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)

    # Calculate the number of batches for validation
    # dataset_size = sum(1 for _ in dataset)
    # val_size = int(dataset_size * validation_split)
    
    

    # Split the dataset
    val_dataset = dataset.take(5)
    train_dataset = dataset.skip(val_size)

    # Prepare the datasets for training
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=20,
        decay_rate=0.50,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="mse")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    class MLflowCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metrics(logs, step=epoch)

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

def create_hist_dataset(hist_list: list, labels_path: str=LABELS_PATH, header_path: str=HEADER_PATH) -> tf.data.Dataset:
    
    logging.basicConfig(filename=f"crate_dataset.log" ,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def combine_lists(a: list, b: list):
        for array in b:
            assert isinstance(array, np.ndarray), f"type of histogram is not numpy nd array: {type(list)}"
        return a + b
    
    client = storage.Client()
    bucket = client.get_bucket(BUCKET) 
    
    hist_name_base = f"{HIST_DEST_PREFIX}"
    histograms = []
    labels = []

    filtered_combinations = []
    processed_county_year = set()
    
    label_df = get_labels(labels_path, header_path)
    
    zeros = 0
    skip = 0
    for combination in hist_list:
        county, fips, year = combination
        
        try:
            label = float(label_df.loc[(label_df["county_name"] == county.upper()) &
                                       (label_df["year"] == year) &
                                       (label_df["state_ansi"] == fips), "target"].iloc[0])
        except IndexError:
            logging.info(f"County {county.upper()} in {year} does not exist in ground truth data. This histogram will be ignored")
            skip += 1
            continue

        zero_count = 0
        hist_by_year = []
        for month in MONTHS:
            file_name = f"{hist_name_base}/{NUM_BINS}_bcukets/{SCALE}/{county.capitalize()}_{fips}/{year}/{month}-{month+1}.npy"
            hist_blob = bucket.blob(file_name)

            if hist_blob.exists():
                content = hist_blob.download_as_bytes()
                binary_data = io.BytesIO(content)
                array = np.load(binary_data)
            else:
                logging.info(f"County {county.upper()}_{fips} in {year} and month {month} does not exist in the histogram set. Zero will be used insted")
                array = np.zeros(NUM_BINS * NUM_BANDS)
                zero_count += 1


            hist_by_year.append(array)
        
        if zero_count > 1:
            skip += 1
            continue
        
        zeros += zero_count
        histograms = combine_lists(histograms, hist_by_year)
        labels.append(label)
            
    # Convert lists to numpy arrays
    histograms_np = np.array(histograms, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)
    
    # Reshape the histograms array
    reshaped_histograms = histograms_np.reshape(-1, len(MONTHS), NUM_BINS * NUM_BANDS)
    
    # Ensure labels are 2D
    labels_np = labels_np.reshape(-1, 1)
    
    print(f"Reshaped histograms shape: {reshaped_histograms.shape}")
    print(f"Labels shape: {labels_np.shape}")
    print(f"Number of filtered combinations: {skip}")
    print(f"Number of missing histograms replaced by zeros: {zeros}")
    
    assert reshaped_histograms.shape[0] == len(hist_list) - skip, "Something went wrong when aggregating training data"
    
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
    
    
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(features, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(features).numpy()])),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label.numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_dataset_to_gcp(dataset, bucket_name='vgnn', file_name='hist_dataset_medium.tfrecords'):
    # Initialize GCP client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a local temporary file
    local_file_name = 'temp_' + file_name
    
    # Write the dataset to the local file
    with tf.io.TFRecordWriter(local_file_name) as writer:
        for features, label in dataset:
            example = serialize_example(features, label)
            writer.write(example)
    
    # Upload the local file to GCS
    blob = bucket.blob(f"dataset/{file_name}")
    blob.upload_from_filename(local_file_name)

    # Remove the local temporary file
    os.remove(local_file_name)

    print(f"Dataset saved to gs://{bucket_name}/dataset/{file_name}")
    
def parse_tfrecord_fn(example_proto):
    # Define the features dictionary that matches the structure used when saving
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    
    # Parse the input tf.Example proto using the feature description
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the feature from the parsed example
    feature = tf.io.parse_tensor(parsed_features['feature'], out_type=tf.float32)
    label = parsed_features['label']
    
    return feature, label

def load_dataset_from_gcp(bucket_name='vgnn', file_name='hist_dataset_medium.tfrecords'):
    # Construct the full GCS path
    gcs_path = f"gs://{bucket_name}/dataset/{file_name}"
    
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(gcs_path)
    
    # Parse the TFRecords
    parsed_dataset = dataset.map(parse_tfrecord_fn)
    
    return parsed_dataset     
    
if __name__ == "__main__":
    dataset = load_dataset_from_gcp()
import os
import sys

import numpy as np
import tensorflow as tf

# Add the necessary directories to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
code_base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(code_base_dir)

# Dataset
from data.test_train_split import test_train_split_multi_modal
from models.lstm import Embeddings, LstmWeather, PreprocessingHead

# Handle TensorFlow imports based on version
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")

# Import Adam optimizer and plot_model based on TensorFlow version
try:
    # For TensorFlow 2.x
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model
except ImportError:
    try:
        # For older TensorFlow versions
        from tensorflow.keras.optimizers.legacy import Adam
        from tensorflow.keras.utils import plot_model
    except ImportError:
        try:
            # For even older versions
            from keras.optimizers import Adam
            from keras.utils import plot_model
        except ImportError:
            print(
                "Could not import Adam optimizer or plot_model. Using default optimizer."
            )
            Adam = None
            plot_model = None

# Make sure eager execution is enabled
tf.config.run_functions_eagerly(True)

# Weather and data paths
WEATHER_PATH = r"C:\Users\tskor\Documents\GitHub\inovation_project\4_Data_Sample\weather_data-adams-2017-CO-8001.csv"
HISTOGRAM_PATH = r"C:\Users\tskor\Documents\GitHub\inovation_project\4_Data_Sample\histogram-08001-2017_reduced.csv"
LABELS_PATH = r"C:\Users\tskor\Documents\GitHub\inovation_project\4_Data_Sample\combined_labels_with_fips.npy"

CAT_FEATURES = ["fips"]

# Get the data
datasets_monthly_train, datasets_monthly_test = test_train_split_multi_modal(
    WEATHER_PATH,
    HISTOGRAM_PATH,
    LABELS_PATH,
    index=["fips", "year"],
    months=[5, 7, 9],
)
histograms_train, weather_train, labels_train = datasets_monthly_train

# Print data information for debugging
print("\nData types and shapes:")
print(
    f"Histograms train type: {type(histograms_train)}, length: {len(histograms_train)}"
)
print(f"Weather train type: {type(weather_train)}, length: {len(weather_train)}")
print(
    f"Labels train type: {type(labels_train)}, shape: {labels_train.shape if hasattr(labels_train, 'shape') else 'No shape'}"
)

# Print the actual content of labels_train to understand its structure
print(
    "Labels train content (first row):",
    labels_train[0] if len(labels_train) > 0 else "Empty",
)

# # Extract label values correctly based on the structure
# if labels_train.shape[1] == 1:
#     # If labels_train has only one column, use that directly
#     labels_values = np.array(labels_train[:, 0], dtype=np.float32)
# elif labels_train.shape[1] >= 3:
#     # If labels_train has 3 or more columns (fips, year, label), use the third column
#     labels_values = np.array(labels_train[:, 2], dtype=np.float32)
# else:
#     # If we're not sure, flatten the array and convert to float
#     labels_values = np.array(labels_train.flatten(), dtype=np.float32)

# print(f"Converted labels shape: {labels_values.shape}")
labels_values = labels_train


# Create a fixed version of the LstmWeather class
class FixedLstmWeather(LstmWeather):
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

        print("shapes")
        print("processed_weather_data: ", processed_weather_data[0].shape)
        print("processed_satellite_data: ", processed_satellite_data[0].shape)

        weather_embeddings = [
            embedding(data)
            for embedding, data in zip(self.weather_embeddings, processed_weather_data)
        ]
        print("weather embeddings: ", weather_embeddings[0].shape)

        satellite_embeddings = [
            embedding(data)
            for embedding, data in zip(
                self.satellite_embeddings, processed_satellite_data
            )
        ]
        print("satellite embeddings: ", satellite_embeddings[0].shape)

        weather_and_satellite_embeddings = [
            weather_and_sat_embedding([weather_embeddings[i], satellite_embeddings[i]])
            for i, weather_and_sat_embedding in enumerate(
                self.concatenate_weather_and_sat_embeddings
            )
        ]
        print("weather sat length: ", len(weather_and_satellite_embeddings))
        print("weather sat shape: ", weather_and_satellite_embeddings[0].shape)

        # Fix the shape issue by ensuring all dimensions are integers
        lstm_input = tf.stack(weather_and_satellite_embeddings, axis=0)
        lstm_input = tf.transpose(lstm_input, perm=[1, 0, 2])

        # # Get the shape and ensure all dimensions are integers
        # shape = lstm_input.shape.as_list()
        # for i in range(len(shape)):
        #     if shape[i] is None:
        #         shape[i] = tf.shape(lstm_input)[i]
        #     elif isinstance(shape[i], float):
        #         shape[i] = int(shape[i])

        # # Reshape to ensure proper shape
        # lstm_input = tf.reshape(lstm_input, shape)

        print("lstm input shape: ", lstm_input.shape)

        # Use try-except to catch and handle any shape errors
        try:
            lstm_output = self.lstm(lstm_input)
            print("lstm_output shape: ", lstm_output.shape)

            output = self.dense(lstm_output)
            print("output shape: ", output.shape)

            return output
        except Exception as e:
            print(f"Error in LSTM processing: {e}")

            # Fallback approach: use a dense layer directly on the concatenated embeddings
            print("Using fallback approach...")

            # Flatten and concatenate all embeddings
            flattened = [
                tf.reshape(emb, [tf.shape(emb)[0], -1])
                for emb in weather_and_satellite_embeddings
            ]
            concatenated = tf.concat(flattened, axis=1)

            # Use a dense layer for prediction
            output = self.dense(concatenated)
            print("fallback output shape: ", output.shape)

            return output


# Create the model with error handling
try:
    print("\nCreating fixed model...")
    model = LstmWeather(weather_train, histograms_train, CAT_FEATURES)
    print("Fixed model created successfully")

    # Compile the model
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss="mse")
    print("Model compiled successfully")

    # Try to call the model directly first to debug
    print("\nTrying direct model call...")
    output = model((weather_train, histograms_train))
    print(f"Model call output shape: {output.shape}")

    try:
        model.fit(x=(weather_train, histograms_train), y=labels_values, epochs=2)
    except Exception as e:
        print(f"Error in model training: {e}")

        # # Implement custom training loop
        # print("\nImplementing custom training loop...")

        # # Get trainable variables
        # trainable_vars = model.trainable_variables

        # # Define loss function
        # loss_fn = tf.keras.losses.MeanSquaredError()

        # # Custom training loop
        # for epoch in range(1):
        #     print(f"Epoch {epoch+1}/10")

        #     # Forward pass
        #     with tf.GradientTape() as tape:
        #         predictions = model((weather_train, histograms_train))
        #         loss_value = loss_fn(labels_values, predictions)

        #     # Compute gradients
        #     gradients = tape.gradient(loss_value, trainable_vars)

        #     # Apply gradients
        #     optimizer.apply_gradients(zip(gradients, trainable_vars))

        #     print(f"  Loss: {loss_value.numpy()}")
        # print("build in method fit exited with error: ", e)
        # print("Custom training completed successfully")

except Exception as e:
    print(f"\nError during model creation or training: {e}")
    import traceback

    traceback.print_exc()

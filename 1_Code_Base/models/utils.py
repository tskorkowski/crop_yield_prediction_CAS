import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf


def inspect_weight_statistics(model):
    stats = {
        "model_name": model.name,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),  # Added timestamp here
        "layers": [],
    }

    for layer in model.layers:
        if layer.weights:
            layer_stats = {
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "weights": [],
            }

            for weight in layer.weights:
                values = weight.numpy()
                weight_stats = {
                    "weight_name": weight.name,
                    "shape": list(
                        values.shape
                    ),  # Convert shape to list for JSON serialization
                    "statistics": {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "quartiles": [
                            float(np.percentile(values, 25)),
                            float(np.percentile(values, 75)),
                        ],
                    },
                }

                # Infer initialization method
                init_method = "unknown"
                if np.abs(np.mean(values)) < 0.1:
                    if 0.4 < np.std(values) < 0.8:
                        init_method = "possibly_glorot"
                    elif np.std(values) < 0.4:
                        init_method = "possibly_he"
                    elif np.abs(np.std(values) - 1.0) < 0.1:
                        init_method = "possibly_orthogonal"
                elif np.all(np.abs(values) < 1e-6):
                    init_method = "zeros"
                elif np.all(np.abs(values - 1.0) < 1e-6):
                    init_method = "ones"

                weight_stats["likely_initialization"] = init_method
                layer_stats["weights"].append(weight_stats)

            stats["layers"].append(layer_stats)

    return stats


def save_weight_statistics(model, output_dir="weight_stats"):
    stats = inspect_weight_statistics(model)

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use timestamp from stats
    filename = f"{output_dir}/weight_stats_{model.name}_{stats['timestamp']}.json"
    with open(filename, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Weight statistics saved to: {filename}")
    return stats


def load_weight_statistics(filename):
    with open(filename, "r") as f:
        stats = json.load(f)
    return stats


def print_weight_stats_summary(stats):
    print(f"Model: {stats['model_name']}")
    for layer in stats["layers"]:
        print(f"\nLayer: {layer['layer_name']} ({layer['layer_type']})")
        for weight in layer["weights"]:
            print(f"  Weight: {weight['weight_name']}")
            print(f"    Shape: {weight['shape']}")
            print(f"    Likely initialization: {weight['likely_initialization']}")
            print(f"    Mean: {weight['statistics']['mean']:.6f}")
            print(f"    Std:  {weight['statistics']['std']:.6f}")


# Custom loss functions


def pen_low_lenient_high_loss(
    y_true, y_pred, low_yield_threshold=80.0, high_yield_threshold=200.0
):
    """
    Custom loss function that focuses on recognizing low crop yield.

    Args:
        y_true: Tensor of true crop yield values.
        y_pred: Tensor of predicted crop yield values.
        low_yield_threshold: Threshold below which yields are considered low.

    Returns:
        loss: Computed loss value.
    """
    # Calculate the absolute error
    squared_error = tf.square(y_true - y_pred)

    # Define weights based on whether the true yield is below the threshold
    weights = tf.where(
        y_true < low_yield_threshold, 10.0, 1.0
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true > high_yield_threshold, 0.5, weights
    )  # Lenient penalty for high yields

    # Compute weighted absolute error
    weighted_squared_error = weights * squared_error

    # You can choose to use Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    loss = tf.reduce_mean(weighted_squared_error)

    return loss


def pen_low_loss(y_true, y_pred, low_yield_threshold=100.0, high_yield_threshold=180.0):
    """
    Custom loss function that focuses on recognizing low crop yield.

    Args:
        y_true: Tensor of true crop yield values.
        y_pred: Tensor of predicted crop yield values.
        low_yield_threshold: Threshold below which yields are considered low.

    Returns:
        loss: Computed loss value.
    """
    # Calculate the absolute error
    squared_error = tf.square(y_true - y_pred)

    # Define weights based on whether the true yield is below the threshold
    weights = tf.where(
        y_true < low_yield_threshold / 2, 8.0, 5.0
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true >= low_yield_threshold, 1.0, weights
    )  # Heavier penalty for low yields
    weights = tf.where(
        y_true > high_yield_threshold, 0.9, weights
    )  # Lenient penalty for high yields

    # Compute weighted absolute error
    weighted_squared_error = weights * squared_error

    # You can choose to use Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    loss = tf.reduce_mean(weighted_squared_error)

    return loss


def model_save(model):

    save_dir = os.path.join("models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"{model.model_name}.keras"))

    return None

import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_PATH = r"C:\Users\tskor\Documents\GitHub\inovation_project\2_gc-pipeline"  # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)


def plot_training_progress(history, naive_loss=None):
    epochs = len(history.history["loss"])

    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    if naive_loss is not None:
        epochs = len(history.history["loss"])
        plt.hlines(
            naive_loss,
            label="Naive Loss",
            xmin=0,
            xmax=epochs - 1,
            colors="r",
            linestyles="dashed",
        )

    plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(
    #     history.history["val_loss"], label="Validation Loss"    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.tight_layout()
    plt.show()

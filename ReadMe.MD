# Crop Yield Prediction

## How to run scripts
Due to incompability between GDAL package and Python and Keras distributions there are two envinronments required to run the scripts.

#### 0_Data
Please use conda envinronment to install the packages according to the requirements_conda.txt

#### 1_Model
Please use pip envinronment to install the packages according to the requirements_pip.txt

## Training dashboard and visualisation
MLFlow Dashboard visualization, run the following command to start the server:
    mlflow ui --port 5000


## Discusion points
1. Which dataset should we use in training?
    CropNet/tiny-CropNet?

    https://huggingface.co/datasets/CropNet/CropNet
    https://huggingface.co/datasets/fudong03/Tiny-CropNet

2. Qualitative results:
    - How do the models perform on various types of crops. Like typical soy/corn vs trees like oranges or avocados or nuts
    - How data hubgry the models are - how much does the performance suffer depending on the amount of training data
    - What is the impact of MODIS vs SENTINEL-2 data?




# Crop Yield Prediction

## How to run scripts
Due to incompability between GDAL package and Python and Keras distributions there are two envinronments required to run the scripts.

Use conda environment to install GDAL and run histograms.py script to generate histograms for the training data.
For the rest of the scripts you can use pip environment as defined in the requirements_pip.txt

#### 0_Data

Does not require conda - see how to run scripts section

1. data_preprocess_sentinel - work in progress - collect information from the google earth engine and return image/or histograms to the google cloud bucket
1. data_preprocess_counties - no started yet - scrip to collect and potentially transform the data
1. USDA folder - groud trouth of the data per county

##### Code based on the previous work

May require conda envinronemt - see how to run scripts

2. Pull MODIS images using pull_modis.py script.
3. Generate histograms for the training data using histograms.py script.
4. Generate datasets using make_datasets.py script.

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




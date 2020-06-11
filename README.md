# Robustness Verification with Non-Uniform Randomized Smoothing

This repository is the official implementation of Robustness Verification with Non-Uniform Randomized Smoothing. 

## Requirements and Setup

We recommend [installing Anaconda](https://docs.anaconda.com/anaconda/install/) for this code. This project was developed on Conda 4.8.3 and Python 3.8.3. To create a new environment and install requirements:

```setup
conda create --name myenv non_uniform_randsmooth
conda activate non_uniform_randsmooth
conda install tensorboard scipy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c anaconda statsmodels
conda install -c conda-forge matplotlib
```

If you wish to run on ImageNet, first obtain a copy of the ILSVRC 2012 challenge dataset and place the image folders in the "datsets/imagenet" directory. (You can change the directory in the code by searching for "CUSTOM" and modifying the filepath in "datasets.py".) Alternatively, you can follow the instructions of [Salman et. al.](https://github.com/Hadisalman/smoothing-adversarial) for running on ImageNet and and change out the appropriate functions by modifying the parts labeled "CUSTOM" in "certify.py" and "datasets.py".

All models and saved sigma vectors we used for the results in our paper are committed to this repository, with the exception of the pre-trained models provided by [Salman et. al.](https://github.com/Hadisalman/smoothing-adversarial). If you wish to use those, please follow their instructions to download their pre-trained models and place the un-zipped models folder into our "models" directory.

## Training and Testing

### certify.py

### create_clean_acc_data.py

### mnist_train.py

## Plotting

### create_cert_acc_plot.py

### create_clean_acc_plot.py

## Reproducing Our Results

Other than our own trained MNIST models, all pre-trained models came from [Salman et. al.](https://github.com/Hadisalman/smoothing-adversarial). Before attempting to re-produce our paper results please follow the instructions in the "Requirements and Setup" section to download the appropriate models and datasets.

## Contributing

TODO: Liscense?

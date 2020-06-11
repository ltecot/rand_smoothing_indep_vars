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

## Training, Testing, and Plotting

The following files are the ones intended for direct usage. Generally all information about how to use the file and possible command inputs are documented in the files themselves, through the parser arguments in the main function and function documentation comments. However, here we'll outline any additional specifics of the files, and give example usages.

Furthermore, note that whenever any custom changes may needed to be made by the user to the code, the appropraite sections will usually be marked by the CUSTOM comment. So it is useful to search the files for this comment marker if you wish to make modifications. 

Note that the only current options for the "model" argument in each of these files is "mnist", "fashion_mnist", "cifar10", "cifar10_robust", "imagenet", and "imagenet_robust". If you wish to add any models (or simply change one of the options we have already written), you will need to search for this CUSTOM comment and change the appropriate sections in the "certify.py" and "datasets.py" files. All other files only need to be modified if you are using only them specifically.

### certify.py

This file will optimize a sigma vector of a smoothed classifer and test for certified area plus accuracy metrics.

Two arguments of note are "sigma_mod" and "save_sigma". Instead of simply optimizing for "epochs" epochs, "sigma_mod" will re-initialize sigma after "sub-epoch" epochs and add "sigma_add" to the new sigma initialization. It repeats this process for "epoch" times. This was used to create the clean accuracy vs. certified area plots. "save_sigma" will save the sigma vector after every epoch to file in the "models/sigmas" folder. The filename will be denoted by the run model, learning rate, gamma, and current epoch. If you wish to further customize the filenames to make them easier to find, you can set the "comment_add" argument.

TODO: Example

### create_clean_acc_data.py

This file tests the original randomized smoothing method. It is similar to certify.py, except it only runs the testing portion and will add "sigma_add" to the used sigma after every epoch. This file was used to 

### mnist_train.py

### create_cert_acc_plot.py

### create_clean_acc_plot.py

## Reproducing Our Results

Other than our own trained MNIST models, all pre-trained models came from [Salman et. al.](https://github.com/Hadisalman/smoothing-adversarial). Before attempting to re-produce our paper results please follow the instructions in the "Requirements and Setup" section to download the appropriate models and datasets.

To generate all the data, run the following commands. This will generate all data neccisary for the plots, as well as sigma images.

```
python certify.py --model=mnist -lr=0.001 --gamma=0.5 --batch_size=128 --test_batch_size=128 --sigma=0.1 --sigma_add=0.1 --epochs=20 --sub_epochs=5 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
python certify.py --model=fashion_mnist -lr=0.001 --gamma=0.5 --batch_size=128 --test_batch_size=128 --sigma=0.1 --sigma_add=0.1 --epochs=20 --sub_epochs=5 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
python certify.py --model=cifar10 -lr=0.0002 --gamma=0.5 --batch_size=16 --test_batch_size=16 --sigma=0.05 --sigma_add=0.05 --epochs=20 --sub_epochs=2 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
python certify.py --model=cifar10_robust -lr=0.0002 --gamma=0.5 --batch_size=16 --test_batch_size=16 --sigma=0.05 --sigma_add=0.05 --epochs=20 --sub_epochs=2 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
python certify.py --model=imagenet -lr=0.0002 --gamma=0.5 --batch_size=1 --test_batch_size=1 --sigma=0.05 --sigma_add=0.05 --epochs=20 --sub_epochs=2 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
python certify.py --model=imagenet_robust -lr=0.0002 --gamma=0.5 --batch_size=1 --test_batch_size=1 --sigma=0.05 --sigma_add=0.05 --epochs=20 --sub_epochs=2 --sigma_mod --save_sigma --batch_smooth=64 --N0=64 --N=512 --N_train=64 --alpha=0.001
```

## Contributing

TODO: Liscense?

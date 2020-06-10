# My Paper Title

This repository is the official implementation of Robustness Verification with Non-Uniform Randomized Smoothing. 

## Requirements

We recommend [installing Anaconda](https://docs.anaconda.com/anaconda/install/) for this code. This project was developed on Conda 4.8.3 and Python 3.8.3. To create a new environment and install requirements:

```setup
conda create --name myenv non_uniform_randsmooth
conda activate non_uniform_randsmooth
conda install tensorboard scipy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c anaconda statsmodels
conda install -c conda-forge matplotlib
```

## Training and Testing

TODO: Explain mnist train, certify, and original smoothing files. Give example commands.

## Plotting

TODO: Explain plotting files. Give example commands

## Reproducing Our Results

Other than our own trained MNIST models, all pre-trained models came from [Salman et. al.](https://github.com/Hadisalman/smoothing-adversarial#getting-started). All their models can be accessed [here](https://drive.google.com/file/d/1GH7OeKUzOGiouKhendC-501pLYNRvU8c/view), however, all pre-trained models that we used are included for use in this repository.

TODO: Commands to reproduce results, and instructions on converting files to be plotted.

TODO: Also include images and plot figures?

## Contributing

TODO: Liscense?

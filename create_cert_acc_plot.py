
from certify import load_dataset, load_model, calculate_objective, get_dataset_name
from mnist_train import Net
from smoothing import Smooth
from datasets import get_input_dim, get_num_classes

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import pickle

# CUSTOM: Add an option for your model or change for existing model.
def get_sigma_vects(model):
    if model == "mnist":
        path1 = 'models/sigmas/sigma_MODEL_mnist_OBJECTIVE_certified_area_LR_0.001_GAMMA_0.5_SIGMA_MOD_EPOCH_19.pt'
        path2 = 'models/sigmas/sigma_MODEL_mnist_OBJECTIVE_certified_area_LR_0.001_GAMMA_0.5_SIGMA_MOD_EPOCH_39.pt'
        return {"Nonisotropic $(\sigma = 0.8)$": torch.load(path1), "Nonisotropic $(\sigma = 1.2)$": torch.load(path2)}
    if model == "fashion_mnist":
        path1 = 'models/sigmas/sigma_MODEL_fashion_mnist_OBJECTIVE_certified_area_LR_0.001_GAMMA_0.5_SIGMA_MOD_EPOCH_34.pt'
        path2 = 'models/sigmas/sigma_MODEL_fashion_mnist_OBJECTIVE_certified_area_LR_0.001_GAMMA_0.5_SIGMA_MOD_EPOCH_74.pt'
        return {"Nonisotropic $(\sigma = 0.7)$": torch.load(path1), "Nonisotropic $(\sigma = 1.5)$": torch.load(path2)}
    elif model == "cifar10":  # R6 - 3 and 5
        path1 = 'models/sigmas/sigma_MODEL_cifar10_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_3.pt'
        path2 = 'models/sigmas/sigma_MODEL_cifar10_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_5.pt'
        return {"Nonisotropic $(\sigma = 0.2)$": torch.load(path1), "Nonisotropic $(\sigma = 0.3)$": torch.load(path2)}
    elif model == "cifar10_robust":  # R6 - 5 and 7
        path1 = 'models/sigmas/sigma_MODEL_cifar10_robust_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_5.pt'
        path2 = 'models/sigmas/sigma_MODEL_cifar10_robust_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_7.pt'
        return {"Nonisotropic $(\sigma = 0.3)$": torch.load(path1), "Nonisotropic $(\sigma = 0.4)$": torch.load(path2)}
    elif model == "imagenet":  # R6 - 4 and 7?
        path1 = 'models/sigmas/sigma_MODEL_imagenet_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_4.pt'
        path2 = 'models/sigmas/sigma_MODEL_imagenet_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_7.pt'
        return {"Nonisotropic $(\sigma = 0.3)$": torch.load(path1), "Nonisotropic $(\sigma = 0.4)$": torch.load(path2)}
    elif model == "imagenet_robust":  # R6 - 5? and 7 
        path1 = 'models/sigmas/sigma_MODEL_imagenet_robust_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_5.pt'
        path2 = 'models/sigmas/sigma_MODEL_imagenet_robust_OBJECTIVE_certified_area_LR_0.0002_GAMMA_0.5_SIGMA_MOD_R6_EPOCH_7.pt'
        return {"Nonisotropic $(\sigma = 0.3)$": torch.load(path1), "Nonisotropic $(\sigma = 0.4)$": torch.load(path2)}

# CUSTOM: Add an option for your model or change for existing model.
def get_sigma_vals(model):
    if model == "mnist":
        return {"Isotropic $(\sigma = 0.8)$": 0.8, "Isotropic $(\sigma = 1.2)$": 1.2}
    elif model == "fashion_mnist":
        return {"Isotropic $(\sigma = 0.6)$": 0.6, "Isotropic $(\sigma = 1.4)$": 1.4}
    elif model == "cifar10":
        return {"Isotropic $(\sigma = 0.17)$": 0.17, "Isotropic $(\sigma = 0.23)$": 0.23}
    elif model == "cifar10_robust":
        return {"Isotropic $(\sigma = 0.22)$": 0.22, "Isotropic $(\sigma = 0.35)$": 0.35}
    elif model == "imagenet":
        return {"Isotropic $(\sigma = 0.26)$": 0.26, "Isotropic $(\sigma = 0.36)$": 0.36}
    elif model == "imagenet_robust":
        return {"Isotropic $(\sigma = 0.28)$": 0.28, "Isotropic $(\sigma = 0.37)$": 0.37}

def load_pickle(args):
    if args.tempload:
        with open(args.temp_pickle, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def write_pickle(args, pkl):
    if args.tempsave:
        with open(args.temp_pickle, 'wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)

def calculate_test_set_objective(args, smoothed_classifier, device, test_loader):
    objectives = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                if prediction != target[i]:
                    objectives.append(float("-inf"))
                else:
                    objectives.append(calculate_objective(True, args.objective, smoothed_classifier.sigma, icdf_pabar).item())
    return sorted(objectives)

def plot_sigma_line(args, model, sig_name, sigma, device, test_loader, pkl, fmt):
    reload_list = []  # CUSTOM: Add sigma names here if you want to reload even when using save file.
    if args.tempload and sig_name in pkl and sig_name not in reload_list:
        objectives = pkl[sig_name][0]
        accuracy = pkl[sig_name][1]
    else:
        smoother = Smooth(model, sigma=sigma, 
                          num_classes=get_num_classes(get_dataset_name(args.model)), 
                          data_shape=get_input_dim(get_dataset_name(args.model)))
        objectives = calculate_test_set_objective(args, smoother, device, test_loader)
        accuracy = np.linspace(1.0, 0.0, num=len(objectives))
        while objectives[0] == float("-inf"):
            objectives = objectives[1:]
            accuracy = accuracy[1:]
        if args.tempsave:
            pkl[sig_name] = [objectives, accuracy]
    plt.plot(objectives, accuracy, fmt, label=sig_name)

def main():
    parser = argparse.ArgumentParser(description='Calculate randomized smoothing certified areas')
    parser.add_argument('--model', type=str,
                        help='filepath to saved model parameters')
    parser.add_argument('--tempsave', action='store_true', default=False,
                        help='save plots to be quick re-loaded')
    parser.add_argument('--tempload', action='store_true', default=False,
                        help='reload any saved plots from pickle file')
    parser.add_argument('--temp_pickle', type=str, default="figures/tempdata.pkl",
                        help='pickle file to save and/or load from')
    parser.add_argument("--batch-smooth", type=int, default=100, help="batch size")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
    parser.add_argument("--N-train", type=int, default=100, help="number of samples to use in training")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Not important for this, ignore')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help='Not important for this, ignore')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader = load_dataset(args, use_cuda)    
    model = load_model(args.model, device)
    model.eval()
    sigma_vects = get_sigma_vects(args.model, args.dataset)
    sigma_vals = get_sigma_vals(args.model)

    pkl = load_pickle(args)
    translate_dict = {}  # CUSTOM: Add key-val pairs to translate key sigma name to val sigma name
    old_pkl = pkl.copy()
    for sig_name in old_pkl:
        if sig_name in translate_dict:
            pkl[translate_dict[sig_name]] = pkl[sig_name]
    vec_fmts = ['-r', '-m']  # CUSTOM: Change or add format strings for non-isotropic plot lines
    val_fmts = ['--b', '--c']  # CUSTOM: Change or add format strings for isotropic plot lines
    with torch.no_grad():
        i = 0
        for sig_name, sigma in sigma_vects.items():
            print("Plotting " + sig_name)
            plot_sigma_line(args, model, sig_name, sigma, device, test_loader, pkl, vec_fmts[i])
            i += 1
            print("Plotted " + sig_name)
        i = 0
        for sig_name, sigma in sigma_vals.items():
            print("Plotting " + sig_name)
            plot_sigma_line(args, model, sig_name, sigma, device, test_loader, pkl, val_fmts[i])
            i += 1
            print("Plotted " + sig_name)
    write_pickle(args, pkl)

    plt.grid()
    plt.ylabel("Certified Accuracy", fontsize=15)
    plt.yticks(fontsize=12)
    plt.xlabel('Certified Area', fontsize=15)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    # CUSTOM: add or change options to customize different plots
    if args.model == "mnist":
        plt.xlim(-4000, 1000)
        plt.ylim(0, 1)
    elif args.model == "fashion_mnist":
        plt.xlim(-4000, 1000)
        plt.ylim(0, 0.85)
    elif args.model == "cifar10":
        plt.xlim(-23000, 0)
        plt.ylim(0, 0.9)
    elif args.model == "cifar10_robust":
        plt.xlim(-15000, 0)
        plt.ylim(0, 0.7)
    elif args.model == "imagenet":
        plt.xlim(-500000, 0)
        plt.ylim(0, 0.85)
    elif args.model == "imagenet_robust":
        plt.xlim(-500000, 0)
        plt.ylim(0, 0.7)

    plt.savefig('figures/cert_acc_' + args.objective + '_' + args.model + '.png')

if __name__ == '__main__':
    main()

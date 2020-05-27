# TODO: Take a model, calculate objective value for each test dataset point, and just create plot from that.

from certify import load_dataset, load_model, calculate_objective

from mnist_train import Net
from smoothing import Smooth
from datasets import get_input_dim, get_num_classes
from architectures import get_architecture

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
# from torch.utils.tensorboard import SummaryWriter

# from math import exp

parser = argparse.ArgumentParser(description='Optimize and compare certified radii')

parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--objective', type=str, default="certified_area")
# parser.add_argument('--indep-vars', action='store_true', default=False,
#                     help='to use indep vars or not')
# parser.add_argument('--create-tradeoff-plot', action='store_true', default=False,
#                     help='forgo optimization and produce plot where lambda is automatically varied')
# parser.add_argument("--lmbd", type=float, default=100000000000, help="tradeoff between accuracy and robust objective")
# parser.add_argument("--lmbd-div", type=float, default=100, help="divider of lambda used when creating tradeoff plots")

parser.add_argument("--batch-smooth", type=int, default=1000, help="batch size")
parser.add_argument("--N0", type=int, default=100) # 100
parser.add_argument("--N", type=int, default=1000, help="number of samples to use") # 100000
parser.add_argument("--N-train", type=int, default=100, help="number of samples to use in training")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
# This sigma is also used as the minimum sigma in the min sigma objective
# parser.add_argument("--sigma", type=float, default=0.5, help="failure probability")
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='Not important for this, ignore')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', # 1000
                    help='Not important for this, ignore')
# parser.add_argument('--epochs', type=int, default=20, metavar='N',
#                     help='number of epochs to train (default: 14)')
# parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
#                     help='learning rate (default: 1.0)')
# parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                     help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save-sigma', action='store_true', default=False,
#                     help='Save the sigma vector')
# parser.add_argument('--gpu', type=int, default=0,
#                     help='The gpu number you are running on.')
args = parser.parse_args()

# Return the objective value for each data point in the test set.
def calculate_test_set_objective(args, model, smoothed_classifier, device, test_loader):
    model.eval()
    objectives = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                # print(icdf_pabar)
                if prediction != target[i]:  # If wrong objective is always not added to cert. acc., so use -inf
                    objectives.append(float("-inf"))
                else:
                    objectives.append(calculate_objective(True, args.objective, smoothed_classifier.sigma, icdf_pabar).item())
    return sorted(objectives)

# def get_models(dataset, device):
#     if dataset == "cifar10":
#         model_reg = load_model("cifar10", device)
#         model_robust = load_model("cifar10_robust", device)
#         return {"Regular Model": model_reg, "Robust Model": model_robust}
#         # return [model_reg, model_robust]
#     # elif dataset == "imagenet":
#     #     pass
#     else:
#         raise Exception("Must enter a valid dataset name")

# Load sigma vectors for when models with a lambda are trained. Placeholder for now.
def get_sigma_vects(model, dataset):
    # Load sigmas
    if model == "mnist":
        path1 = 'models/sigmas/sigma_MODEL_mnist_OBJECTIVE_certified_area_MULTIPLE_SIGMA_TRADEOFF_PLOT_LAMBDA_10000.0'
        path2 = 'models/sigmas/sigma_MODEL_mnist_OBJECTIVE_certified_area_MULTIPLE_SIGMA_TRADEOFF_PLOT_LAMBDA_1e-06'
        return {"$\lambda = 10^4$": torch.load(path1), "$\lambda = 10^{-6}$": torch.load(path2)}
    elif model == "cifar10":
        return {}
        # return {"$\lambda$ = 0.12": 0.12 * torch.ones(get_input_dim(dataset)), "$\lambda$ = 1.00": 1.0 * torch.ones(get_input_dim(dataset))}
    elif model == "cifar10_robust":
        return {}
        # return {"$\lambda$ = 0.12": 0.12 * torch.ones(get_input_dim(dataset)), "$\lambda$ = 1.00": 1.0 * torch.ones(get_input_dim(dataset))}

# Load sigma values for original method testing.
def get_sigma_vals(model):
    if model == "mnist":
        return {"$\sigma$ = 0.6": 0.7, "$\sigma$ = 1.4": 1.40}
    elif model == "cifar10":
        return {"$\sigma$ = 0.12": 0.12, "$\sigma$ = 1.00": 1.00}
    elif model == "cifar10_robust":
        return {"$\sigma$ = 0.12": 0.12, "$\sigma$ = 1.00": 1.00}

# Plots a line for a smoother defined by the sigma
def plot_sigma_line(args, model, sig_name, sigma, device, test_loader):
    smoother = Smooth(model, num_classes=get_num_classes(args.dataset), sigma=sigma, indep_vars=True, data_shape=get_input_dim(args.dataset))
    objectives = calculate_test_set_objective(args, model, smoother, device, test_loader)
    accuracy = np.linspace(1.0, 0.0, num=len(objectives))
    # print(objectives)
    # print(accuracy)
    while objectives[0] == float("-inf"):
        objectives = objectives[1:]
        accuracy = accuracy[1:]
    # objectives = [exp(obj) for obj in objectives]
    plt.plot(objectives, accuracy, label=sig_name)

def main():
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader = load_dataset(args, use_cuda)    
    model = load_model(args.model, device)
    sigma_vects = get_sigma_vects(args.model, args.dataset)
    sigma_vals = get_sigma_vals(args.model)
    # sigma_vals = {}

    with torch.no_grad():
        for sig_name, sigma in sigma_vects.items():
            print("Plotting " + sig_name)
            plot_sigma_line(args, model, sig_name, sigma, device, test_loader)
            print("Plotted " + sig_name)
        for sig_name, sigma in sigma_vals.items():
            print("Plotting " + sig_name)
            plot_sigma_line(args, model, sig_name, sigma, device, test_loader)
            print("Plotted " + sig_name)
        
    # plt.style.use('seaborn-darkgrid')
    plt.grid()
    plt.ylabel("Certified Accuracy")
    plt.legend()
    # Plot Title
    if args.model == "mnist":
        plt.title("Mnist Model")
    elif args.model == "cifar10":
        plt.title("Cifar10 Model")
    elif args.model == "cifar10":
        plt.title("Cifar10 Model")
    elif args.model == "cifar10_robust":
        plt.title("Cifar10 Robust Model")
    # Plot X axis
    if args.objective == "certified_area":
        plt.xlabel("Certified Area")
        # plt.xlabel("Certified Area (Log)")
    elif args.objective == "largest_delta_norm":
        plt.xlabel("Maximum Pertubation")
    
    plt.savefig('figures/cert_acc_' + args.objective + '_' + args.model + '.png')

if __name__ == '__main__':
    main()

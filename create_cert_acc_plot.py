# TODO: Take a model, calculate objective value for each test dataset point, and just create plot from that.

from certify import load_dataset, load_model, calculate_objective

from mnist_train import Net
from smoothing import Smooth
from datasets import get_dataset, get_input_dim, get_num_classes
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
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Optimize and compare certified radii')

# parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--objective', type=str, default="")
parser.add_argument('--indep-vars', action='store_true', default=False,
                    help='to use indep vars or not')
parser.add_argument('--create-tradeoff-plot', action='store_true', default=False,
                    help='forgo optimization and produce plot where lambda is automatically varied')
parser.add_argument("--lmbd", type=float, default=100000000000, help="tradeoff between accuracy and robust objective")
parser.add_argument("--lmbd-div", type=float, default=100, help="divider of lambda used when creating tradeoff plots")

parser.add_argument("--batch-smooth", type=int, default=1000, help="batch size")
parser.add_argument("--N0", type=int, default=100) # 100
parser.add_argument("--N", type=int, default=1000, help="number of samples to use") # 100000
parser.add_argument("--N-train", type=int, default=100, help="number of samples to use in training")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
# This sigma is also used as the minimum sigma in the min sigma objective
# parser.add_argument("--sigma", type=float, default=0.5, help="failure probability")
# parser.add_argument('--batch-size', type=int, default=8, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', # 1000
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=20, metavar='N',
#                     help='number of epochs to train (default: 14)')
# parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
#                     help='learning rate (default: 1.0)')
# parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                     help='Learning rate step gamma (default: 0.7)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--save-sigma', action='store_true', default=False,
#                     help='Save the sigma vector')
# parser.add_argument('--gpu', type=int, default=0,
#                     help='The gpu number you are running on.')

# Return the objective value for each data point in the test set.
def calculate_test_set_objective(args, model, smoothed_classifier, device, test_loader, epoch, lmbd):
    model.eval()
    objectives = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                if prediction == target[i]:  # If wrong objective is always not added to cert. acc., so use -inf
                    objectives.append(float("-inf"))
                else:
                    objectives.append(calculate_objective(args.indep_vars, args.objective, smoothed_classifier.sigma, icdf_pabar).item())
    return sort(objectives)

def get_models(dataset, device):
    if dataset == "cifar10":
        pass
    elif dataset == "imagenet":
        pass
    else:
        raise Exception("Must enter a valid dataset name")

def main():
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, test_loader = load_dataset(args, use_cuda)    
    models = get_models(args.dataset, device)
    for model in models:
        smoother = Smooth(model, num_classes=get_num_classes(args.dataset), sigma=args.sigma, indep_vars=args.indep_vars, data_shape=get_input_dim(args.dataset))
    # optimizer = optim.Adadelta([smoother.sigma], lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

if __name__ == '__main__':
    main()

# Does runs on original random smoothing to produce a plot for clean acc vs. cert area by varying the sigma.

from __future__ import print_function

from certify import load_dataset, load_model, calculate_objective
from mnist_train import Net
from smoothing import Smooth
from datasets import get_dataset, imagenet_trainset, get_input_dim, get_num_classes
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

def test(args, model, smoothed_classifier, device, test_loader, epoch, lmbd, writer, comment, sigma):
    model.eval()
    area_objective = 0
    norm_objective = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                    accuracy += 1
                    area_objective += calculate_objective(True, "certified_area", smoothed_classifier.sigma, icdf_pabar)
                    norm_objective += calculate_objective(True, "largest_delta_norm", smoothed_classifier.sigma, icdf_pabar)
        area_objective /= accuracy  # Want to average objectives that are actually certified
        norm_objective /= accuracy
        accuracy /= len(test_loader.dataset)
        print('\nAverage Area objective: {:.4f}'.format(area_objective))
        print('Average Norm objective: {:.4f}'.format(norm_objective))
        print('Percent correct: {:.4f}'.format(accuracy))
        writer.add_scalar('orig_rand_smooth_plot/area_objective', area_objective, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/norm_objective', norm_objective, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/accuracy', accuracy, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/sigma', sigma, epoch-1)
    return lmbd

def main():

    parser = argparse.ArgumentParser(description='Optimize and compare certified radii')

    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)  # TODO: Refactor out. Always determined by model anyways.
    parser.add_argument("--sigma", type=float, default=2, help="tradeoff between accuracy and robust objective")
    parser.add_argument("--sigma_sub", type=float, default=0.1, help="divider of lambda used when creating tradeoff plots")
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',  # TODO: combine batch sizes, should be same basically
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', # 1000
                        help='input batch size for testing (default: 1000)')

    parser.add_argument("--batch-smooth", type=int, default=100, help="batch size")
    parser.add_argument("--N0", type=int, default=100) # 100
    parser.add_argument("--N", type=int, default=1000, help="number of samples to use") # 100000
    parser.add_argument("--N-train", type=int, default=100, help="number of samples to use in training")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--epochs', type=int, default=21, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    comment = '_ORIG_RANDSMOOTH_PLOT_MODEL_' + args.model

    writer = SummaryWriter(comment=comment)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = load_dataset(args, use_cuda)    
    model = load_model(args.model, device)

    sigma = args.sigma
    for epoch in range(1, args.epochs + 1):
        smoother = Smooth(model, num_classes=get_num_classes(args.dataset), sigma=sigma, indep_vars=True, data_shape=get_input_dim(args.dataset))
        test(args, model, smoother, device, test_loader, epoch, 0, writer, comment, sigma)
        sigma = sigma - args.sigma_sub

    writer.close()

if __name__ == '__main__':
    main()




# To gather certified area data for isotropic randomized smoothing.

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
from torch.utils.tensorboard import SummaryWriter

def test(args, smoothed_classifier, device, test_loader, epoch, writer):
    """Tests smoothed classifer for certified area objective, writes results to tensorboard.
    Args:
        args (argparse.ArgumentParser): Arguments containing N0, N, alpha, and batch_smooth.
        smoothed_classifier (smoothing.Smooth): Randomized smoother.
        device (torch.device): Device to load the data onto.
        test_loader (torch.utils.data.DataLoader): Test dataset loader.
        epoch (int): Epoch of optimization process.
        writer (torch.utils.tensorboard.SummaryWriter): Writer for tensorboard data.
    """
    area_objective = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                    accuracy += 1
                    area_objective += calculate_objective(smoothed_classifier.sigma, icdf_pabar)
        area_objective /= accuracy  # Want to average objectives that are actually certified
        accuracy /= len(test_loader.dataset)
        print('\nAverage Area objective: {:.4f}'.format(area_objective))
        print('Percent correct: {:.4f}'.format(accuracy))
        writer.add_scalar('orig_rand_smooth_plot/area_objective', area_objective, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/accuracy', accuracy, epoch-1)

def main():
    parser = argparse.ArgumentParser(description='get clean accuracy vs certified area plots for original randomized smoothing')
    parser.add_argument('--model', type=str,
                        help='filepath to saved model parameters')
    parser.add_argument("--sigma", type=float, default=0.025, 
                        help="constant elements in sigma vector are initialized to")
    parser.add_argument("--sigma_add", type=float, default=0.025, 
                        help="amount to add to sigma per epoch")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='input batch size for testing')
    parser.add_argument("--batch_smooth", type=int, default=64, 
                        help="batch size for smoothed classifer when sampling random noise")
    parser.add_argument("--N0", type=int, default=64,
                        help='number of samples used in when estimating smoothed classifer prediction')
    parser.add_argument("--N", type=int, default=512, 
                        help="number of samples used when estimating paBar")
    parser.add_argument("--N_train", type=int, default=64, 
                        help="number of samples to use in training when the smoothed classifer samples noise")
    parser.add_argument("--alpha", type=float, default=0.001, 
                        help="probability that paBar is not a true lower bound on pA")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    args = parser.parse_args()

    comment = '_ORIG_RANDSMOOTH_PLOT_MODEL_' + args.model
    writer = SummaryWriter(comment=comment)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = load_dataset(get_dataset_name(args.model), args.batch_size, args.test_batch_size, use_cuda)   
    model = load_model(args.model, device)
    model.eval()

    sigma = args.sigma
    for epoch in range(1, args.epochs + 1):
        smoother = Smooth(model, sigma=sigma, 
                          num_classes=get_num_classes(get_dataset_name(args.model)), 
                          data_shape=get_input_dim(get_dataset_name(args.model)))
        writer.add_scalar('orig_rand_smooth_plot/sigma', sigma, epoch-1)
        test(args, smoother, device, test_loader, epoch, writer)
        sigma = sigma + args.sigma_add

    writer.close()

if __name__ == '__main__':
    main()




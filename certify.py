# Test to maximize the "certification radius" of both smoothing methods.
# Similar to mnist_train but modified to optimize sigmas instead.

from __future__ import print_function

from mnist_train import Net
from smoothing import Smooth
from datasets import get_dataset, get_input_dim
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

# Slope of hinge loss to enforce min sigma objective.
MIN_SIGMA_HINGE_SLOPE = 10000000000

parser = argparse.ArgumentParser(description='Optimize and compare certified radii')

parser.add_argument('--model', type=str)
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
parser.add_argument("--sigma", type=float, default=0.5, help="failure probability")
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N', # 1000
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--save-model', action='store_true', default=True,
#                     help='For Saving the current Model')
# parser.add_argument('--gpu', type=int, default=0,
#                     help='The gpu number you are running on.')


args = parser.parse_args()
comment = '_MODEL_' + args.model
comment = comment + '_OBJECTIVE_' + args.objective + '_MULTIPLE_SIGMA' if args.indep_vars else comment + '_SINGLE_SIGMA'
if args.create_tradeoff_plot:
    comment = comment + '_TRADEOFF_PLOT'
writer = SummaryWriter(comment=comment)
# GLOBAL_LMBD = args.lmbd  # So it can be varied by the plotting procedure

def load_dataset(dataset_name, use_cuda):
    if dataset_name == "mnist":
        kwargs = {'pin_memory': True} if use_cuda else {}  # 'num_workers': 1, 
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)  # Smoothing only can handle one at a time anyways right now
            # batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif dataset_name == "cifar10":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(get_dataset("cifar10", "train"), batch_size=args.test_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(get_dataset("cifar10", "test"), batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # elif dataset_name == "imagenet": # Needs some extra file stuff before this will work
    else:
        raise Exception("Must enter a valid dataset name")
    return train_loader, test_loader

def load_model(model_name, device):
    if model_name == "mnist":
        model = Net().to(device)
        model.load_state_dict(torch.load('mnist_cnn.pt'))
    elif model_name == "cifar10":
        checkpoint = torch.load("models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/checkpoint.pth.tar")
        model = get_architecture(checkpoint["arch"], "cifar10")
        model.load_state_dict(checkpoint['state_dict'])
    elif model_name == "cifar10_robust":
        checkpoint = torch.load("models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/checkpoint.pth.tar")
        model = get_architecture(checkpoint["arch"], "cifar10")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise Exception("Must enter a valid model name")
    return model

# Calculate proper objective we are trying to maximize
def calculate_objective(args, sigma, icdf_pabar):
    # if torch.sum(sigma <= 0):
    #     print(sigma)
    #     print(icdf_pabar)
    if not args.indep_vars:
        objective = sigma * icdf_pabar  # Just do certified radius
    else:
        if args.objective == "largest_delta_norm":
            objective = torch.norm(sigma, p=2) * icdf_pabar
        elif args.objective == "minimum_sigma_largest_delta_norm":
            objective = torch.norm(sigma, p=2) * icdf_pabar + MIN_SIGMA_HINGE_SLOPE * torch.sum(torch.min(sigma - args.sigma, torch.tensor([0.]).cuda()))
        elif args.objective == "certified_area":
            sigma = torch.abs(sigma)  # For log calculation. Negative or positive makes no difference in our formulation.
            eps = 0.000000001 # To prevent log from returning infinity.
            if not torch.is_tensor(icdf_pabar):
                icdf_pabar = torch.tensor(icdf_pabar.item())
            objective = torch.sum(torch.log(sigma+eps)) + torch.numel(sigma) * torch.log(icdf_pabar+eps)  # sum of log of simgas + d * log of inverse CDF of paBar
        else:
            raise Exception("Must enter a valid objective")
    # print(objective)
    return objective

# def calculate_loss(objective_value, ce_loss, lambda_param):
#     # lambda_param * F.cross_entropy(model_output, true_class) + objective_value
#     lambda_param * F.cross_entropy(model_output, true_class) + objective_value

def train(args, model, smoothed_classifier, device, train_loader, optimizer, epoch, lmbd):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        ce_loss = torch.tensor([0.0]).cuda()
        objective = torch.tensor([0.0]).cuda()
        accuracy = 0
        # avg_icdf = 0
        # print(ce_loss)
        # print(objective)
        for i in range(data.shape[0]):
            prediction, icdf_pabar, smoothed_output = smoothed_classifier.certify_training(data[i], args.N0, args.N_train, args.alpha, args.batch_smooth, target[i])
            # print(icdf_pabar)
            # print(smoothed_output.unsqueeze(0).shape)
            # print(target[i:i+1].shape)
            ce_loss += F.cross_entropy(smoothed_output.unsqueeze(0), target[i:i+1])
            if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                accuracy += 1
                objective += calculate_objective(args, smoothed_classifier.sigma, icdf_pabar)
        ce_loss /= data.shape[0]
        objective /= data.shape[0]
        accuracy /= data.shape[0]
        loss = lmbd * ce_loss - objective
        loss.backward()
        optimizer.step()
        # # Doesn't actually make a difference in our version, with exception of area calculation.
        # smoothed_classifier.sigma = torch.abs(smoothed_classifier.sigma)  # Enforces positive-only sigma
        
        # print(ce_loss)
        # print(objective)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCE_Loss: {:.6f}\tObjective: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), ce_loss.item(), objective.item()))
            print('Accuracy: {:.5f} \t sigma mean: {:.5f} \t sigma stddev: {:.5f}'.format(
                accuracy, 
                torch.abs(smoothed_classifier.sigma).mean().item(),
                torch.abs(smoothed_classifier.sigma).std().item()))
    # Write last objective, loss, and accuracy to tensorboard
    writer.add_scalar('ce_loss/train', ce_loss, epoch-1)
    writer.add_scalar('objective/train', objective, epoch-1)
    writer.add_scalar('accuracy/train', accuracy, epoch-1)

# TODO: Record and report test accuracy of the smoothed model too.
def test(args, model, smoothed_classifier, device, test_loader, epoch, lmbd):
    model.eval()
    # test_loss = 0
    objective = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(data.shape)
            # print(target.shape)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                # test_loss += radius
                if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                    accuracy += 1
                    objective += calculate_objective(args, smoothed_classifier.sigma, icdf_pabar)
        # test_loss /= len(test_loader.dataset)
        objective /= len(test_loader.dataset)
        accuracy /= len(test_loader.dataset)
        print('\nAverage Test objective: {:.4f}'.format(objective))
        print('Percent correct: {:.4f}'.format(accuracy))
        print('Sigma avg: {:.4f}\n'.format(torch.abs(smoothed_classifier.sigma).mean()))
        # print('Sigma:')
        # print(smoothed_classifier.sigma)
        # plt.imshow(smoothed_classifier.sigma[0].cpu().numpy())
        # save_image(data[0], 'gen_files/sigma_viz.png')
        writer.add_scalar('objective/test', objective, epoch-1)
        writer.add_scalar('accuracy/test', accuracy, epoch-1)
        writer.add_scalar('sigma/mean', torch.abs(smoothed_classifier.sigma).mean(), epoch-1)
        writer.add_scalar('sigma/stddev', torch.abs(smoothed_classifier.sigma).std(), epoch-1)
        # writer.add_scalar('Percent_Correct', perc_correct, epoch-1)
        if args.indep_vars:
            sigma_img = torch.abs(smoothed_classifier.sigma)  # For image. Negative or positive makes no difference in our formulation.
            sigma_img = sigma_img - sigma_img.min()
            sigma_img = sigma_img / sigma_img.max()
            writer.add_image('sigma', sigma_img, epoch-1)
            # save_image(sigma_img[0], 'gen_files/sigma_viz.png')
            # writer.add_image('Sigma', (data[0] - data[0].min()) / (data[0] - data[0].min()).max(), epoch-1)
        # print(smoothed_classifier.sigma)
        if args.create_tradeoff_plot:  # Keep in mind this will transform the x-axis into ints, so this should not be used for the paper plots.
            writer.add_scalar('tradeoff_plot/lambda', lmbd, epoch-1)
            writer.add_scalar('tradeoff_plot/acc_obj', accuracy, objective)
            # writer.add_scalar('tradeoff_plot/acc_sigma_mean', accuracy, smoothed_classifier.sigma.mean())
            lmbd /= args.lmbd_div
    return lmbd

def main():
    # Training settings
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = load_dataset(args.dataset, use_cuda)    

    model = load_model(args.model, device)

    smoother = Smooth(model, num_classes=10, sigma=args.sigma, indep_vars=args.indep_vars, data_shape=get_input_dim(args.dataset))
    optimizer = optim.Adadelta([smoother.sigma], lr=args.lr)

    lmbd = args.lmbd
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, smoother, device, train_loader, optimizer, epoch, lmbd)
        lmbd = test(args, model, smoother, device, test_loader, epoch, lmbd)
        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()




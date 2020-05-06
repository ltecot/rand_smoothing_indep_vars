# Test to maximize the "certification radius" of both smoothing methods.
# Similar to mnist_train but modified to optimize sigmas instead.

from __future__ import print_function

from mnist_train import Net
from smoothing import Smooth

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
parser.add_argument("--batch-smooth", type=int, default=1000, help="batch size")
parser.add_argument("--N0", type=int, default=100) # 100
parser.add_argument("--N", type=int, default=1000, help="number of samples to use") # 100000
parser.add_argument("--N-train", type=int, default=100, help="number of samples to use in training")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--indep-vars', action='store_true', default=False,
                    help='to use indep vars or not')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', # 1000
#                     help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
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
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')

args = parser.parse_args()
filename_suffix = 'multiple_sigma' if args.indep_vars else 'single_sigma'
writer = SummaryWriter(filename_suffix=filename_suffix)

def load_mnist_model():
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))

def train(args, model, smoothed_classifier, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        avg_radius = 0
        avg_percent = 0
        # avg_icdf = 0
        for i in range(data.shape[0]):
            percent, radius = smoothed_classifier.certify_training(data[i], args.N0, args.N_train, args.alpha, args.batch_smooth, target[i])
            avg_radius += radius
            avg_percent += percent
            # avg_icdf += icdf
        avg_percent /= data.shape[0]
        avg_radius /= data.shape[0]
        # avg_icdf /= data.shape[0]
        # TODO: Change to remove instances where the predicted class is wrong. Maybe ok to keep?
        loss = -avg_radius
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('Percent: {:.5f} \t sigma mean: {:.5f} \t sigma stddev: {:.5f}'.format(
                avg_percent.item(), 
                smoothed_classifier.sigma.mean().item(),
                smoothed_classifier.sigma.std().item()))
    # Write last radius and percent to tensorboard
    writer.add_scalar('Radius/train', avg_radius, epoch-1)
    writer.add_scalar('Percent/train', avg_percent, epoch-1)

def test(args, model, smoothed_classifier, device, test_loader, epoch):
    model.eval()
    # test_loss = 0
    avg_radius = 0
    avg_percent = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(data.shape)
            # print(target.shape)
            prediction, percent, radius = smoothed_classifier.certify(data[0], args.N0, args.N, args.alpha, args.batch_smooth)
            # test_loss += radius
            avg_radius += radius
            avg_percent += percent
        # test_loss /= len(test_loader.dataset)
        avg_radius /= len(test_loader.dataset)
        avg_percent /= len(test_loader.dataset)
        print('\nAverage Test upper bound: {:.4f}'.format(avg_radius))
        print('Sigma avg: {:.4f}\n'.format(smoothed_classifier.sigma.mean()))
        # print('Sigma:')
        # print(smoothed_classifier.sigma)
        # plt.imshow(smoothed_classifier.sigma[0].cpu().numpy())
        # save_image(data[0], 'gen_files/sigma_viz.png')
        writer.add_scalar('Radius/test', avg_radius, epoch-1)
        writer.add_scalar('Percent/test', avg_percent, epoch-1)
        writer.add_scalar('Sigma_Mean', smoothed_classifier.sigma.mean(), epoch-1)
        if args.indep_vars:
            sigma_img = smoothed_classifier.sigma - smoothed_classifier.sigma.min()
            sigma_img = sigma_img / sigma_img.max()
            writer.add_image('Sigma', sigma_img, epoch-1)
            save_image(sigma_img[0], 'gen_files/sigma_viz.png')
            # writer.add_image('Sigma', (data[0] - data[0].min()) / (data[0] - data[0].min()).max(), epoch-1)
        # print(smoothed_classifier.sigma)

def main():
    # Training settings
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
        batch_size=1, shuffle=True, **kwargs)  # Smoothing only can handle one at a time anyways right now
        # batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    smoother = Smooth(model, num_classes=10, sigma=0.1, indep_vars=args.indep_vars, data_shape=[1, 28, 28])
    optimizer = optim.Adadelta([smoother.sigma], lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, smoother, device, train_loader, optimizer, epoch)
        test(args, model, smoother, device, test_loader, epoch)
        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()




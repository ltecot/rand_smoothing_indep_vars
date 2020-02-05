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
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='Optimize and compare certified radii')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, help="noise hyperparameter")
# parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch-smooth", type=int, default=1000, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
# parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# Batch sizes need to be one for now, smoothing class can't handle more at the moment.
parser.add_argument('--batch-size', type=int, default=1, metavar='N',  # 64
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', # 1000
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
# args = parser.parse_args()

args = parser.parse_args()

def load_mnist_model():
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))

def train(args, model, smoothed_classifier, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data[0]
        # print(data.shape)
        # print(target.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # output = model(data)
        # smoothed_classifier = Smooth(base_classifier=model, num_classes=10, sigma=0.01)
        radius, percent = smoothed_classifier.certify_training(data, args.N0, args.N, args.alpha, args.batch_smooth, target)
        # loss = F.nll_loss(output, target)
        # TODO: Change to remove instances where the predicted class is wrong.
        loss = -radius
        # print(radius)
        loss.backward()
        # print(smoothed_classifier.sigma.grad)
        # print(model.parameters)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print("percent: ", percent)
            print("sigma: ", smoothed_classifier.sigma)

def test(args, model, smoothed_classifier, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # output = model(data)
            prediction, radius = smoothed_classifier.certify(data, N0, N, alpha, batch)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += radius

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('\nTest set: Average radius: {:.4f}'.format(test_loss))

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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    smoother = Smooth(model, num_classes=10, sigma=0.1)
    optimizer = optim.Adadelta([smoother.sigma], lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch)
        train(args, model, smoother, device, train_loader, optimizer, epoch)
        test(args, model, smoother, device, test_loader)
        scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()




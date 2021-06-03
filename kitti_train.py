from __future__ import print_function
import argparse
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from certify import load_dataset, load_model

def train(model, device, train_loader, optimizer, epoch, log_interval):
    """Train MNIST model.
    Args:
        model (nn.Module): Model to train.
        device (torch.device): Device to load data onto.
        train_loader (torch.utils.data.DataLoader): Train dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer for model.
        epoch (int): Epoch of optimization process.
        log_interval (int): Interval of epochs to write results to stdout.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    """Test MNIST model.
    Args:
        model (nn.Module): Model to test.
        device (torch.device): Device to load data onto.
        test_loader (torch.utils.data.DataLoader): Test dataset loader.
    """
    model.eval()
    test_loss = 0
    correct = 0
    # targets = [0 for _ in range(10)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # for t in target:
            #     targets[t] += 1
    test_loss /= len(test_loader.dataset)
    # print(targets)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train MNIST models.')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = load_dataset("kitti", args.batch_size, args.test_batch_size, use_cuda)
    model = models.resnet50(pretrained=False).cuda()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader)
        scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "models/kitti.pt")

if __name__ == '__main__':
    main()

# Need above 80%, that's percent of dataset in class 0.
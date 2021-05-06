# To train and test non-isotropic randomized smoothing.

from mnist_train import Net
from smoothing import Smooth
from datasets import get_dataset, imagenet_trainset, get_input_dim, get_num_classes, KittiDataset
from architectures import get_architecture

import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# CUSTOM: Add an option for your model
def get_dataset_name(model):
    """Converts model name to corresponding dataset name.
    Args:
        model (str): Name of model.
    Returns:
        (str) Name of dataset.
    """
    if model == "mnist":
        return "mnist"
    elif model == "fashion_mnist":
        return "fashion_mnist"
    elif model == "cifar10" or model == "cifar10_robust":
        return "cifar10"
    elif model == "imagenet" or model == "imagenet_robust":
        return "imagenet"
    elif model == "kitti":
        return "kitti"

# CUSTOM: Add an option for your model
def load_dataset(dataset, batch_size, test_batch_size, use_cuda):
    """Loads the train and test loader for a dataset.
    Args:
        dataset (str): Name of dataset.
        batch_size (int): Batch size of train set.
        test_batch_size (int): Batch size of test set.
        use_cuda (bool): Use cuda if true.
    Returns:
        (torch.utils.data.DataLoader, torch.utils.data.DataLoader) 
        Train dataset dataloader, test dataset dataloader.
    """
    if dataset == "mnist":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == "fashion_mnist":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('datasets', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('datasets', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=test_batch_size, shuffle=True, **kwargs) 
    elif dataset == "cifar10":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(get_dataset("cifar10", "train"), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(get_dataset("cifar10", "test"), batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == "imagenet":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_set, test_set = imagenet_trainset()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == "kitti":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        data_path = 'data/train/'
        dataset = KittiDataset('../datasets/kitti')
        # print(len(dataset))
        train_set, test_set = torch.utils.data.random_split(dataset, [6733, 748]) # 7481
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        raise Exception("Must enter a valid dataset name")
    return train_loader, test_loader

# CUSTOM: Add an option for your model
def load_model(model_name, device):
    """Loads the model.
    Args:
        model_name (str): Name of model.
        device (torch.device): Device to load the model onto.
    Returns:
        (nn.Module) Loaded Pytorch model.
    """
    if model_name == "mnist":
        model = Net().to(device)
        model.load_state_dict(torch.load('models/mnist_cnn.pt'))
    elif model_name == "fashion_mnist":
        model = Net().to(device)
        model.load_state_dict(torch.load('models/fashion_mnist_cnn.pt'))
    elif model_name == "cifar10":
        checkpoint = torch.load("models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/checkpoint.pth.tar")
        model = get_architecture(checkpoint["arch"], "cifar10")
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(torch.load('models/cifar10.pt'))
    elif model_name == "cifar10_robust":
        checkpoint = torch.load("models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/checkpoint.pth.tar")
        model = get_architecture(checkpoint["arch"], "cifar10")
        model.load_state_dict(checkpoint['state_dict'])
    elif model_name == "imagenet":
        # checkpoint = torch.load("models/pretrained_models/imagenet/PGD_1step/imagenet/eps_127/resnet50/noise_0.25/checkpoint.pth.tar")
        # model = get_architecture(checkpoint["arch"], "imagenet")
        # model.load_state_dict(checkpoint['state_dict'])
        model = models.resnet50(pretrained=True).cuda()
    elif model_name == "imagenet_robust":
        checkpoint = torch.load("models/pretrained_models/imagenet/PGD_1step/imagenet/eps_1024/resnet50/noise_0.25/checkpoint.pth.tar")
        model = get_architecture(checkpoint["arch"], "imagenet")
        model.load_state_dict(checkpoint['state_dict'])
    elif model_name == "kitti":
        model = models.resnet50(pretrained=False).cuda()
        model.load_state_dict(torch.load('models/kitti.pt'))
    else:
        raise Exception("Must enter a valid model name")
    return model

def calculate_objective(sigma, icdf_pabar):
    """Calculates the certified area objective.
    Args:
        sigma (torch.tensor): Sigma vector.
        icdf_pabar (torch.tensor/np.array/float): Inverse Gaussian CDF of paBar.
    Returns:
        (torch.tensor) Certified area objective, single element tensor.
    """
    sigma = torch.abs(sigma)  # For log calculation. Negative or positive makes no difference in our formulation.
    eps = 0.000000001 # To prevent log from returning infinity.
    if isinstance(icdf_pabar, float):
        icdf_pabar = torch.tensor(icdf_pabar)
    elif not torch.is_tensor(icdf_pabar):
        icdf_pabar = torch.tensor(icdf_pabar.item())
    objective = torch.sum(torch.log(sigma+eps)) + torch.numel(sigma) * torch.log(icdf_pabar+eps)
    return objective

def train(args, smoothed_classifier, device, train_loader, optimizer, epoch, writer):
    """Optimizes sigma for the certified area objective, writes results to tensorboard.
    Args:
        args (argparse.ArgumentParser): Arguments containing N_train and batch_smooth.
        smoothed_classifier (smoothing.Smooth): Randomized smoother.
        device (torch.device): Device to load the data onto.
        train_loader (torch.utils.data.DataLoader): Train dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer for smoother's sigma vector.
        epoch (int): Epoch of optimization process.
        writer (torch.utils.tensorboard.SummaryWriter): Writer for tensorboard data.
    """
    avg_objective = torch.tensor([0.0]).cuda()
    avg_accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        objective = torch.tensor([0.0]).cuda()
        accuracy = 0
        for i in range(data.shape[0]):
            prediction, icdf_pabar = smoothed_classifier.certify_training(data[i], args.N_train, args.batch_smooth, target[i])
            if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                accuracy += 1
                objective += calculate_objective(smoothed_classifier.sigma, icdf_pabar)
        avg_objective += objective
        avg_accuracy += accuracy
        if accuracy != 0:
            objective /= accuracy  # Want to average objectives that are actually certified
            accuracy /= data.shape[0]
            loss = -objective
            loss.backward()
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tObjective: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), objective.item()))
            print('Accuracy: {:.5f} \t sigma mean: {:.5f} \t sigma stddev: {:.5f}'.format(
                accuracy, 
                torch.abs(smoothed_classifier.sigma).mean().item(),
                torch.abs(smoothed_classifier.sigma).std().item()))
    # Write avg objective and accuracy to tensorboard
    avg_objective /= avg_accuracy  # Divide by number of correctly classified points
    avg_accuracy /= len(train_loader.dataset)
    writer.add_scalar('objective/train', avg_objective, epoch-1)
    writer.add_scalar('accuracy/train', avg_accuracy, epoch-1)

def test(args, smoothed_classifier, device, test_loader, epoch, writer, comment):
    """Tests smoothed classifer for certified area objective, writes results to tensorboard.
    Args:
        args (argparse.ArgumentParser): Arguments containing N0, N, alpha, and batch_smooth.
        smoothed_classifier (smoothing.Smooth): Randomized smoother.
        device (torch.device): Device to load the data onto.
        test_loader (torch.utils.data.DataLoader): Test dataset loader.
        epoch (int): Epoch of optimization process.
        writer (torch.utils.tensorboard.SummaryWriter): Writer for tensorboard data.
        comment (str): String to append to use for saved sigma filenames.
    """
    objective = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.shape[0]):
                prediction, icdf_pabar = smoothed_classifier.certify(data[i], args.N0, args.N, args.alpha, args.batch_smooth)
                if prediction == target[i]:  # Add 0 to all if it predicts wrong.
                    accuracy += 1
                    objective += calculate_objective(smoothed_classifier.sigma, icdf_pabar)
        objective /= accuracy  # Want to average objectives that are actually certified
        accuracy /= len(test_loader.dataset)
        print('\nAverage Test objective: {:.4f}'.format(objective))
        print('Percent correct: {:.4f}'.format(accuracy))
        print('Sigma avg: {:.4f}\n'.format(torch.abs(smoothed_classifier.sigma).mean()))
        writer.add_scalar('objective/test', objective, epoch-1)
        writer.add_scalar('accuracy/test', accuracy, epoch-1)
        writer.add_scalar('sigma/mean', torch.abs(smoothed_classifier.sigma).mean(), epoch-1)
        writer.add_scalar('sigma/stddev', torch.abs(smoothed_classifier.sigma).std(), epoch-1)
        sigma_img = torch.abs(smoothed_classifier.sigma)  # For image. Negative or positive makes no difference in our formulation.
        sigma_img = sigma_img - sigma_img.min()
        sigma_img = sigma_img / sigma_img.max()
        writer.add_image('sigma_linear_normalized', sigma_img, epoch-1)
        if args.save_sigma:
            torch.save(smoothed_classifier.sigma, 'models/sigmas/sigma' + comment + '_EPOCH_' + str(epoch) + '.pt')

def main():
    parser = argparse.ArgumentParser(description='Optimize certified area')
    parser.add_argument('--model', type=str,
                        help='filepath to saved model parameters')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='learning rate step gamma')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=2,
                        help='input batch size for testing')
    parser.add_argument("--sigma", type=float, default=0.025, 
                        help="constant elements in sigma vector are initialized to")
    parser.add_argument("--sigma_add", type=float, default=0.025, 
                        help="amount to add to sigma per sub-epoch if doing sigma modulation")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--sub_epochs', type=int, default=10,
                        help='number of epochs to optimize for per initialization if doing sigma modulation')
    parser.add_argument('--comment_add', type=str, default="",
                        help='string to add to the end of name used for tensorboard and sigma filesaves')
    parser.add_argument('--sigma_mod', action='store_true', default=True,
                        help='modulate sigma, re-initialize after sub-epochs to train multiple times')
    parser.add_argument('--save_sigma', action='store_true', default=True,
                        help='save the sigma vector')
    parser.add_argument("--batch_smooth", type=int, default=32, 
                        help="batch size for smoothed classifer when sampling random noise")
    parser.add_argument("--N0", type=int, default=32,
                        help='number of samples used in when estimating smoothed classifer prediction')
    parser.add_argument("--N", type=int, default=256,
                        help="number of samples used when estimating paBar")
    parser.add_argument("--N_train", type=int, default=32, 
                        help="number of samples to use in training when the smoothed classifer samples noise")
    parser.add_argument("--alpha", type=float, default=0.001, 
                        help="probability that paBar is not a true lower bound on pA")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    comment = '_MODEL_' + args.model + '_LR_' + str(args.lr) + '_GAMMA_' + str(args.gamma)
    if args.sigma_mod:
        comment = comment + '_SIGMA_MOD'
    comment = comment + args.comment_add

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = load_dataset(get_dataset_name(args.model), args.batch_size, args.test_batch_size, use_cuda)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < 10:
            print(batch_idx, data.size(), target)
    model = load_model(args.model, device)
    model.eval()
    writer = SummaryWriter(comment=comment)
    if args.sigma_mod:
        sigma = args.sigma
        for epoch in range(1, args.epochs + 1):
            smoother = Smooth(model, sigma=sigma, 
                              num_classes=get_num_classes(get_dataset_name(args.model)), 
                              data_shape=get_input_dim(get_dataset_name(args.model)))
            optimizer = optim.Adam([smoother.sigma], lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            for sub_epoch in range(1, args.sub_epochs + 1):
                effective_epoch = (epoch - 1) * args.sub_epochs + sub_epoch
                writer.add_scalar('sigma_mod/sigma_init', sigma, effective_epoch-1)
                train(args, smoother, device, train_loader, optimizer, effective_epoch, writer)
                test(args, smoother, device, test_loader, effective_epoch, writer, comment)
                scheduler.step() 
            sigma += args.sigma_add
    else:
        smoother = Smooth(model, sigma=sigma, 
                          num_classes=get_num_classes(get_dataset_name(args.model)), 
                          data_shape=get_input_dim(get_dataset_name(args.model)))
        optimizer = optim.Adam([smoother.sigma], lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, smoother, device, train_loader, optimizer, epoch, writer)
            test(args, smoother, device, test_loader, epoch, writer, comment)
            scheduler.step()
    writer.close()

if __name__ == '__main__':
    main()




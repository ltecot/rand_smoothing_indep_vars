# Does runs on original rand smoothing to produce a plot for clean acc vs. objective by varying the sigma.

# Test to maximize the "certification radius" of both smoothing methods.
# Similar to mnist_train but modified to optimize sigmas instead.

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
    # test_loss = 0
    area_objective = 0
    norm_objective = 0
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
                    area_objective += calculate_objective(True, "certified_area", smoothed_classifier.sigma, icdf_pabar)
                    norm_objective += calculate_objective(True, "largest_delta_norm", smoothed_classifier.sigma, icdf_pabar)
        # test_loss /= len(test_loader.dataset)
        # objective /= len(test_loader.dataset)
        area_objective /= accuracy  # Want to average objectives that are actually certified
        norm_objective /= accuracy  # Want to average objectives that are actually certified
        accuracy /= len(test_loader.dataset)
        print('\nAverage Area objective: {:.4f}'.format(area_objective))
        print('Average Norm objective: {:.4f}'.format(norm_objective))
        print('Percent correct: {:.4f}'.format(accuracy))
        # print('Sigma avg: {:.4f}\n'.format(torch.abs(smoothed_classifier.sigma).mean()))
        # print('Sigma:')
        # print(smoothed_classifier.sigma)
        # plt.imshow(smoothed_classifier.sigma[0].cpu().numpy())
        # save_image(data[0], 'gen_files/sigma_viz.png')
        writer.add_scalar('orig_rand_smooth_plot/area_objective', area_objective, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/norm_objective', norm_objective, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/accuracy', accuracy, epoch-1)
        writer.add_scalar('orig_rand_smooth_plot/sigma', sigma, epoch-1)
        # writer.add_scalar('sigma/mean', torch.abs(smoothed_classifier.sigma).mean(), epoch-1)
        # writer.add_scalar('sigma/stddev', torch.abs(smoothed_classifier.sigma).std(), epoch-1)
        # writer.add_scalar('Percent_Correct', perc_correct, epoch-1)
        # if args.indep_vars:
        #     # Linear Scaled Image
        #     sigma_img = torch.abs(smoothed_classifier.sigma)  # For image. Negative or positive makes no difference in our formulation.
        #     sigma_img = sigma_img - sigma_img.min()
        #     sigma_img = sigma_img / sigma_img.max()
        #     writer.add_image('sigma_linear_normalized', sigma_img, epoch-1)
        #     # Gaussian scaled image
        #     sigma_gaus_img = torch.abs(smoothed_classifier.sigma)  # For image. Negative or positive makes no difference in our formulation.
        #     sigma_gaus_img = (sigma_gaus_img - torch.mean(sigma_gaus_img)) / torch.std(sigma_gaus_img)
        #     sigma_gaus_img = ((sigma_gaus_img * 0.25) + 0.5)  # Assuming normal dist, will put %95 of values in [0,1] range
        #     sigma_gaus_img = torch.clamp(sigma_gaus_img, 0, 1)  # Clips out of range values
        #     writer.add_image('sigma_gaussian_normalized', sigma_gaus_img, epoch-1)
        #     # save_image(sigma_img[0], 'gen_files/sigma_viz.png')
        #     # writer.add_image('Sigma', (data[0] - data[0].min()) / (data[0] - data[0].min()).max(), epoch-1)
        # # print(smoothed_classifier.sigma)
        # if args.save_sigma:
        #     torch.save(smoothed_classifier.sigma, 'models/sigmas/sigma' + comment + '_LAMBDA_' + str(lmbd) + '.pt')
        # if args.create_tradeoff_plot:  # Keep in mind this will transform the x-axis into ints, so this should not be used for the paper plots.
        #     writer.add_scalar('tradeoff_plot/lambda', lmbd, epoch-1)
        #     # writer.add_scalar('tradeoff_plot/acc_obj', accuracy, objective)
        #     # writer.add_scalar('tradeoff_plot/acc_sigma_mean', accuracy, smoothed_classifier.sigma.mean())
        #     lmbd /= args.lmbd_div
    return lmbd

def main():

    parser = argparse.ArgumentParser(description='Optimize and compare certified radii')

    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)  # TODO: Refactor out. Always determined by model anyways.
    # parser.add_argument('--objective', type=str, default="")
    # parser.add_argument('--create-tradeoff-plot', action='store_true', default=True,
    #                     help='forgo optimization and produce plot where lambda is automatically varied')
    # parser.add_argument('--save-sigma', action='store_true', default=True,
    #                     help='Save the sigma vector')
    parser.add_argument("--sigma", type=float, default=2.0, help="tradeoff between accuracy and robust objective")
    parser.add_argument("--sigma_sub", type=float, default=0.2, help="divider of lambda used when creating tradeoff plots")
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='learning rate (default: 1.0)')

    # parser.add_argument('--indep-vars', action='store_true', default=True,  # TODO: Pretty much always true at this point. Refactor out later.
    #                     help='to use indep vars or not')
    parser.add_argument("--batch-smooth", type=int, default=64, help="batch size")
    parser.add_argument("--N0", type=int, default=64) # 100
    parser.add_argument("--N", type=int, default=512, help="number of samples to use") # 100000
    parser.add_argument("--N-train", type=int, default=64, help="number of samples to use in training")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    # This sigma is also used as the minimum sigma in the min sigma objective
    # parser.add_argument("--sigma", type=float, default=0.1, help="failure probability")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',  # TODO: combine batch sizes, should be same basically
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', # 1000
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--gpu', type=int, default=0,
    #                     help='The gpu number you are running on.')

    args = parser.parse_args()

    comment = '_ORIG_RANDSMOOTH_PLOT_MODEL_' + args.model
    # comment = "Testing"

    writer = SummaryWriter(comment=comment)

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = load_dataset(args, use_cuda)    
    model = load_model(args.model, device)
    # smoother = Smooth(model, num_classes=get_num_classes(args.dataset), sigma=args.sigma, indep_vars=args.indep_vars, data_shape=get_input_dim(args.dataset))
    # optimizer = optim.Adadelta([smoother.sigma], lr=args.lr)
    # optimizer = optim.Adam([smoother.sigma], lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # lmbd = args.lmbd
    sigma = args.sigma
    for epoch in range(1, args.epochs + 1):
        # train(args, model, smoother, device, train_loader, optimizer, epoch, lmbd, writer)
        smoother = Smooth(model, num_classes=get_num_classes(args.dataset), sigma=sigma, indep_vars=True, data_shape=get_input_dim(args.dataset))
        test(args, model, smoother, device, test_loader, epoch, 0, writer, comment, sigma)
        sigma = sigma - args.sigma_sub
        # scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()




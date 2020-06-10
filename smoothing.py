# Original file from https://github.com/locuslab/smoothing/blob/master/code/core.py
# Modified to perform non-isotropic randomized smoothing.

import numpy as np
from math import ceil
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Smooth(object):
    """Provides a smoothed classifer of the provided base model.
    Attributes:
        base_classifier (torch.nn.Module): The base classifer.
        sigma (torch.tensor): Sigma tensor for variances of input.
        num_classes (int): Number of output classes.
        unit_norm (torch.distributions): Unit norm. Intended for interal usage 
        eps (float): 
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier, sigma, num_classes, data_shape=None):
        """
        Args:
            base_classifier (torch.nn.Module): Base classifer for smoothed classifer.
            sigma (torch.tensor / float): Initial sigma. Will use a tensor if input, 
                                          otherwise will be be tensor with this constant value
                                          in all elements.
            num_classes (int): Number of output classes.
            data_shape (iterable[int]): Shape of input data. Only needed if sigma isn't a tensor.
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        if torch.is_tensor(sigma):
            self.sigma = torch.tensor(sigma, dtype=torch.float, device='cuda', requires_grad=True)
        else:
            self.sigma = torch.tensor(sigma * np.ones(data_shape), dtype=torch.float, device='cuda', requires_grad=True)
        self.unit_norm = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        self.eps = 0.0000001 # To prevent icdf from returning infinity.

    def certify(self, x, n0, n, alpha, batch_size):
        """ Monte Carlo algorithm for certifying that g's prediction around x.
        With probability at least 1 - alpha, the class returned by this method will equal g(x).
        The returned icdf_paBar can be used with sigma to determine certified deltas.
        Args:
            x (torch.tensor): the input
            n0 (int): the number of Monte Carlo samples to use for selection
            n (int): the number of Monte Carlo samples to use for estimation
            alpha (float): the failure probability
            batch_size (int): batch size to use when evaluating the base classifier
        Returns:
            (int, float) Predicted class, inverse Gaussian CDF of paBar.
            In the case of abstention, the class will be ABSTAIN and icdf_pabar 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, norm.ppf(pABar)

    def certify_training(self, x, n, batch_size, truth_label):
        """ Softmax approximation of certify, to be used during training of sigmas for gradient information.
        Args:
            x (torch.tensor): The input
            n (int): The number of Monte Carlo samples to use for estimation
            batch_size (int): Batch size to use when evaluating the base classifier
            truth_label (int): True class of input x.
        Returns:
            (int, float) Predicted class, inverse Gaussian CDF of paBar.
            In the case of abstention, the class will be ABSTAIN and icdf_pabar 0.
        """
        self.base_classifier.eval()
        counts_estimation, true_class_softmax_sum = self._sample_noise(x, n, batch_size, training=True, truth_label=truth_label)
        cAHat = counts_estimation.argmax().item()
        pA = true_class_softmax_sum / n
        if pA < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, self.unit_norm.icdf(torch.clamp(pA, self.eps, 1-self.eps))

    def predict(self, x, n, alpha, batch_size):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        Args:
            x (torch.tensor): the input
            n (int): the number of Monte Carlo samples to use for estimation
            alpha (float): the failure probability
            batch_size (int): batch size to use when evaluating the base classifier
        Returns:
            (int) Predicted class. In the case of abstention, the class will be ABSTAIN.
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x, num, batch_size, training=False, truth_label=None):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        Args:
            x (torch.tensor): the input
            num (int): number of samples to collect
            batch_size (int): batch size to use when evaluating the base classifier
            training (bool): Indicates whether training and should return softmax output as well.
            truth_label (int): True label of input. Only needed if training is true.
        Returns:
            (ndarray[int]) an ndarray[int] of length num_classes containing the per-class counts
            (ndarray[int], torch.tensor) if training, one-element tensor containing the summed softmax outputs
                                         of each sample for the truth label class.
        """
        counts = np.zeros(self.num_classes, dtype=int)
        if training:
            true_class_softmax_sum = torch.tensor(0.0).cuda()
            # summed_outputs = torch.zeros(self.num_classes).cuda()  # For cross-entropy loss
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = self.sigma * torch.randn_like(batch)
            output = self.base_classifier(batch + noise)
            predictions = output.argmax(1)
            counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            if training:
                softmax_out = F.softmax(output, dim=1)
                true_class_softmax_sum += torch.sum(softmax_out[:, truth_label])
                # summed_outputs += torch.sum(output, dim=0)
        if training:
            return counts, true_class_softmax_sum
        else:
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]
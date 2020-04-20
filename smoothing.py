# Original file from https://github.com/locuslab/smoothing/blob/master/code/core.py
# TODO: Look at gradients. The _sample_noise function uses argmax, so it might just be that
# the gradient for sigma decreases but is never negative.

import torch
from torch.distributions import Normal
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

# Class copied directly from original paper repo, modified for sigma optimization
class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, indep_vars=False, data_shape=None):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        # self.sigma = sigma
        if indep_vars:
            # self.sigma = torch.ones(data_shape, requires_grad=True, device='cuda')
            # self.sigma = torch.ones(data_shape, requires_grad=True)
            # self.sigma = torch.from_numpy(sigma * np.ones(data_shape)).float().cuda()
            self.sigma = torch.tensor(sigma * np.ones(data_shape), dtype=torch.float, device='cuda', requires_grad=True)
            # print(self.sigma.type())
            # with torch.no_grad():
            #     self.sigma = sigma * self.sigma
        else:
            self.sigma = torch.tensor(sigma, requires_grad=True)
            # self.sigma = torch.tensor(sigma, requires_grad=True, device='cuda')
        self.unit_norm = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.eps = 0.000001 # To prevent icdf from returning infinity.
        self.indep_vars = indep_vars

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
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
            if self.indep_vars:
                radius = torch.norm(self.sigma, p=2) * norm.ppf(pABar)
            else:
                radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    # TODO: Update docs
    # TODO: Maybe should keep in the if statement at the end. Might cause issues with gradient
    #       to cut off variables that go below the threshold
    # Because the real certify percent estimate is bounded in number of samples with this value used
    # in training, should be fine.
    def certify_training(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, truth_label) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        # counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        # cAHat = counts_selection.argmax().item()
        # cAHat = truth_label[0]  # Use training data instead of guessing what should be optimized
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, training=True, truth_label=truth_label)
        # use these samples to estimate a lower bound on pA
        # nA = counts_estimation[cAHat].item()
        # nA = counts_estimation[0]
        # pABar = self._lower_confidence_bound(nA, n, alpha)
        # if pABar < 0.5:
        #     return Smooth.ABSTAIN, 0.0
        # else:
        # radius = self.sigma * norm.ppf(pABar)
        # print(nA, n)
        if self.indep_vars:
            # print(self.sigma.grad)
            radius = torch.norm(self.sigma, p=2).cuda() * self.unit_norm.icdf(torch.abs(counts_estimation / n - self.eps))
        else:
            # print(self.sigma.grad)
            radius = self.sigma * self.unit_norm.icdf(torch.abs(counts_estimation / n - self.eps))
        # return cAHat, radius
        return counts_estimation / n, self.unit_norm.icdf(counts_estimation / n - self.eps), radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
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

    # TODO: Also update docs here, different return behavior for training
    def _sample_noise(self, x: torch.tensor, num: int, batch_size, training=False, truth_label=None) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            if not training:
                counts = np.zeros(self.num_classes, dtype=int)
            else:
                counts = torch.tensor(0.0).cuda()
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                # print(batch.type())
                # noise = torch.randn_like(batch, device='cuda') * self.sigma
                # print(torch.randn_like(batch).get_device())
                # print(self.sigma.get_device())
                noise = torch.randn_like(batch) * self.sigma
                # print(self.sigma.shape)
                # print(batch.shape)
                # print(noise.shape)
                predictions = self.base_classifier(batch + noise).argmax(1)
                if not training:
                    counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
                else:
                    counts += torch.sum(predictions == truth_label)
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
import torch
from torch import nn

from model.likelihood.bernoulli import BernoulliLikelihood
from model.likelihood.normal import DiscretizedNormalLikelihood
from model.likelihood.mixlog import DiscretizedMixtureLogisticLikelihood


def select_likelihood(ll):
    if ll == "binary":
        return BernoulliLikelihood
    elif ll == "discretized_normal":
        return DiscretizedNormalLikelihood
    elif ll == "discretized_mix_logistic":
        return DiscretizedMixtureLogisticLikelihood
    else:
        print("Please choose one between binary, discretized_normal, discretized_mix_logistic")
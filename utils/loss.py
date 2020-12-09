import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = torch.max(x, dim=axis)[0]
    m2 = torch.max(x, dim=axis, keepdims=True)[0]
    return m + torch.log(torch.sum(torch.exp(x-m2), dim=axis))

def discretized_mix_logistic_loss(target, input, log_scales=-14.0, num_classes=256,
                                  rescale_target=1.0, size_average=True, reduce=True):
    
    assert input.shape[2] % 3 == 0
    num_mix = (input.shape[2]) // 3
    logit_probs = input[:, :, : num_mix]
    means = input[:, :, num_mix : 2 * num_mix]
    target = target * torch.ones(1, 1, num_mix)
    centered_x = target - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + rescale_target / (num_classes - 1))
    cdf_plus = nn.Sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - rescale_target / (num_classes - 1))
    cdf_min = nn.Sigmoid(min_in)

    # Log probability for edge case of min value (before scaling).
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # Log probability for edge case of max value (before scaling).
    log_one_minus_cdf_min = -F.softplus(min_in)
    # Probability for all other cases.
    cdf_delta = cdf_plus - cdf_min
    # Log probability in the center of the bin, to be used in extreme cases.
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # Width in x axis for calculating log probability in the center of the bin.

    x_width = (num_classes - 1.0) / (2 * rescale_target)
    inner_inner_cond = torch.FloatTensor(cdf_delta > 1e-5)  
    inner_inner_out = (
        inner_inner_cond * torch.log(torch.maximum(cdf_delta, 1e-12)) +
        (1. - inner_inner_cond) * (log_pdf_mid - np.log(x_width)))
    inner_cond = torch.FloatTensor(target > 0.999 * rescale_target)
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = torch.FloatTensor(target < -0.999 * rescale_target)
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + nn.LogSoftmax(logit_probs, dim=-1)
    if not reduce:
        return -log_sum_exp(log_probs)
    if size_average:
        return -torch.mean(log_sum_exp(log_probs), dim=-1)

    return -torch.sum(log_sum_exp(log_probs), dim=-1)

def sample_from_discretized_mix_logistic(input, log_scales=-14.0, rescale_target=1.0, random_boundary=1e-2):

    assert input.shape[2] % 3 == 0
    num_mix = (input.shape[2]) // 3
    logit_probs = input[:, :, :num_mix]
    u = (random_boundary - (1. - random_boundary)) * torch.rand(logit_probs.shape) + (1. - random_boundary)
    x = logit_probs - torch.log(-torch.log(u))
    max_indices = torch.argmax(x, dim=-1)
    one_hot = F.one_hot(max_indices, num_mix).to(float)

    means = torch.sum(input[:, :, num_mix : 2 * num_mix] * one_hot, dim=-1)
    log_scales = torch.maximum(
            torch.sum(input[:, :, num_mix * 2 : num_mix *3 ] * one_hot, dim=-1), log_scales)
  
    u = (random_boundary - (1. - random_boundary)) * torch.rand(means.shape) + (1. - random_boundary)
    predict = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    predict = predict / rescale_target

    x = torch.unsqueeze(torch.minimum(torch.maximum(predict, -0.999), 0.999), axis=-1)

    return x

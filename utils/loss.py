import numpy as np
import torch
import torch.nn as nn

def discretized_mix_logistic_loss(target, input, log_scales=-14.0, num_classes=256,
                                  rescale_target=1.0, size_average=True, reduce=True):
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
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # Log probability for edge case of max value (before scaling).
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    # Probability for all other cases.
    cdf_delta = cdf_plus - cdf_min
    # Log probability in the center of the bin, to be used in extreme cases.
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    # Width in x axis for calculating log probability in the center of the bin.

    x_width = (num_classes - 1.0) / (2 * rescale_target)
    inner_inner_cond = tf.to_float(cdf_delta > 1e-5)  
    inner_inner_out = (
        inner_inner_cond * tf.log(tf.maximum(cdf_delta, 1e-12)) +
        (1. - inner_inner_cond) * (log_pdf_mid - np.log(x_width)))
    inner_cond = tf.to_float(target > 0.999 * rescale_target)
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = tf.to_float(target < -0.999 * rescale_target)
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + tf.nn.log_softmax(logit_probs, axis=-1)
    if not reduce:
        return -log_sum_exp(log_probs)
    if size_average:
        return -tf.reduce_mean(log_sum_exp(log_probs), [-1])

    return -tf.reduce_sum(log_sum_exp(log_probs), [-1])
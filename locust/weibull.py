import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import weibull_min
from scipy.special import gamma

PATH = "./locust/"

def truncated_weibull(shape_k, scale_lambda, tmin, tmax, size=10000):
    """Generate samples from a truncated Weibull distribution."""
    def cdf(x):
        return 1 - np.exp(- (x / scale_lambda) ** shape_k)

    def ppf(u):
        return scale_lambda * (-np.log(1 - u)) ** (1 / shape_k)
    
    u = np.random.uniform(0, 1, size)
    cdf_min = cdf(tmin)
    cdf_max = cdf(tmax)
    u_truncated = cdf_min + u * (cdf_max - cdf_min)
    samples = ppf(u_truncated)
    return samples

def compute_weibull_scale(desired_mean, shape_k):
    return desired_mean / gamma(1 + 1 / shape_k)

def get_n_weibull_variables(shape_k, scale_lambda, tmin, tmax, size, seed):
    np.random.seed(seed) 
    samples = truncated_weibull(shape_k, scale_lambda, tmin, tmax, size)
    
    return samples


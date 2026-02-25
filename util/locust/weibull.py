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

def plot(tmin, tmax, shape_k, scale_lambda, N, T, stages, offset, seed, label="tmp"):
        # Plot histogram of samples
        plt.figure(figsize=(10, 6))
        bins = np.linspace(tmin, tmax, 100)
        plt.hist([int(e["load"]) - o for e, o in zip(stages, offset)], bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label="Truncated Weibull Samples")

        # Plot analytical truncated PDF
        x_vals = np.linspace(tmin, tmax, 500)
        pdf_vals = weibull_min.pdf(x_vals, c=shape_k, scale=scale_lambda)
        cdf_min = weibull_min.cdf(tmin, c=shape_k, scale=scale_lambda)
        cdf_max = weibull_min.cdf(tmax, c=shape_k, scale=scale_lambda)
        truncated_pdf = pdf_vals / (cdf_max - cdf_min)

        plt.plot(x_vals, truncated_pdf, 'r--', lw=2, label="Truncated Weibull PDF")
        plt.title("Truncated Weibull Distribution Samples")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

        os.makedirs(f"./output/{label}/plots", exist_ok=True)
        plt.savefig(f"./output/{label}/plots/user_count_distribution_{seed}_{label}.png")
        plt.clf()

        # plot users per second
        print(0, len(stages), [int(e["load"]) for e in stages])

        load = [int(e["load"]) for e in stages]
        print(load)

        x = np.linspace(0, T,len(stages))
        plt.plot(x, load)
        plt.title("Truncated Weibull Load generation in Users per Second.")
        plt.xlabel("time [S]")
        plt.ylabel("users [#]")
        plt.grid(True)

        plt.savefig(f"./output/{label}/plots/users_over_time_{seed}_{label}.png")
        plt.clf()
   
if __name__ == "__main__":
    import os
    import math
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-shape", type=int, required=False, default=1)
    parser.add_argument("--w-mean", type=int, required=False, default=1200)
    parser.add_argument("--w-user-min", type=int, required=False, default=1000)
    parser.add_argument("--w-user-max", type=int, required=False, default=6000)
    parser.add_argument("--w-dt", type=int, required=False, default=20)
    parser.add_argument("--w-ls-y", type=int, required=False, default=3500)
    parser.add_argument("--run-time", type=int, required=False, default=60*60)
    args = parser.parse_args()

    w_shape = args.w_shape
    w_mean = args.w_mean
    U_min = args.w_user_min
    U_max = args.w_user_max
    t = args.run_time
    dt = args.w_dt
    ls_y = args.w_ls_y
    label = "tmp"
    
    os.makedirs(f"./output/tmp/plots", exist_ok=True)
    stages = []
    ## magic-kit
    _lambda = compute_weibull_scale(w_mean, w_shape)
    N = int(t/dt)
    L = get_n_weibull_variables(w_shape, _lambda, U_min, U_max, N, 42)

    l_prev = 0


    # or logspace()
    offset = np.linspace(U_min, ls_y, N)

    for s, l, o in zip(range(dt,t+dt,dt), L, offset):
        stages.append({"start": s, "load": int(l + o), "rate": int(math.ceil(abs((l_prev - l)/dt)))})
        l_prev = l
    

    # plot(U_min, U_max, w_shape, _lambda, N, t, stages, offset, 42,  label)
    print(stages)


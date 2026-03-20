import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(1)

# generate Pareto sample
def generate_pareto(alpha, x_m, n):
    return x_m * (1 + np.random.pareto(alpha, size=n))

# MOM estimator of alpha
def pareto_mom_alpha(sample):
    x_min = np.min(sample)
    x_bar = np.mean(sample)
    return x_bar / (x_bar - x_min)

# simulate many alpha-hats
def simulate_alpha_hats(alpha, x_m, n, n_sim=3000):
    alpha_hats = np.empty(n_sim)

    for i in range(n_sim):
        sample = generate_pareto(alpha, x_m, n)
        alpha_hats[i] = pareto_mom_alpha(sample)

    return alpha_hats

# check asymptotic normality
def check_asymptotic_normality(alpha, x_m, sample_sizes, n_sim=3000):
    for n in sample_sizes:
        alpha_hats = simulate_alpha_hats(alpha, x_m, n, n_sim=n_sim)

        # normalized estimator
        z = np.sqrt(n) * (alpha_hats - alpha)

        print(f"n = {n}")
        print(f"mean(alpha_hat) = {np.mean(alpha_hats):.6f}")
        print(f"std(alpha_hat)  = {np.std(alpha_hats, ddof=1):.6f}")
        print("-" * 50)

        # histogram
        plt.figure(figsize=(8, 4))
        plt.hist(z, bins=50, density=True)
        plt.title(f"Histogram of sqrt(n)(alpha_hat - alpha), n={n}")
        plt.xlabel("sqrt(n)(alpha_hat - alpha)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

        # QQ-plot
        plt.figure(figsize=(6, 6))
        stats.probplot(z, dist="norm", plot=plt)
        plt.title(f"QQ-plot of sqrt(n)(alpha_hat - alpha), n={n}")
        plt.grid(True)
        plt.show()

# good case
check_asymptotic_normality(alpha=3.0, x_m=2.0, sample_sizes=[100, 200, 500, 1000, 20000], n_sim=3000)

# bad case
check_asymptotic_normality(alpha=1.5, x_m=2.0, sample_sizes=[100, 200, 500, 1000], n_sim=3000)
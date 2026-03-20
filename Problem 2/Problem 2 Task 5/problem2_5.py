import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(1)

def generate_pareto(alpha, x_m, n):
    # generate Pareto sample
    return x_m * (1 + np.random.pareto(alpha, size=n))


def pareto_mle(sample):
    # MLE for alpha and x_m
    x_min = np.min(sample)
    alpha_hat = len(sample) / np.sum(np.log(sample / x_min))
    x_m_hat = x_min
    return alpha_hat, x_m_hat

def simulate_mle(alpha, x_m, n, n_sim=3000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # save all estimates
    alpha_hats = np.empty(n_sim)
    xm_hats = np.empty(n_sim)

    for i in range(n_sim):
        sample = generate_pareto(alpha, x_m, n)
        alpha_hat, xm_hat = pareto_mle(sample)
        alpha_hats[i] = alpha_hat
        xm_hats[i] = xm_hat

    return alpha_hats, xm_hats


def check_alpha_mle_normality(alpha, x_m, sample_sizes, n_sim=3000):
    for n in sample_sizes:
        alpha_hats, xm_hats = simulate_mle(alpha, x_m, n, n_sim=n_sim)

        # normalized version
        z = np.sqrt(n) * (alpha_hats - alpha)

        print(f"n = {n}")
        print(f"mean(alpha_hat_MLE) = {np.mean(alpha_hats):.6f}")
        print(f"std(alpha_hat_MLE)  = {np.std(alpha_hats, ddof=1):.6f}")
        print(f"mean(x_m_hat_MLE)   = {np.mean(xm_hats):.6f}")
        print(f"std(x_m_hat_MLE)    = {np.std(xm_hats, ddof=1):.6f}")
        print("-" * 55)

        # histogram
        plt.figure(figsize=(8, 4))
        plt.hist(z, bins=50, density=True)
        plt.title(f"Histogram of sqrt(n)(alpha_hat_MLE - alpha), n={n}")
        plt.xlabel("sqrt(n)(alpha_hat_MLE - alpha)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

        # QQ plot
        plt.figure(figsize=(6, 6))
        stats.probplot(z, dist="norm", plot=plt)
        plt.title(f"QQ-plot of sqrt(n)(alpha_hat_MLE - alpha), n={n}")
        plt.grid(True)
        plt.show()


# check normality for alpha hat
print("=== Asymptotic normality check for alpha_hat_MLE ===")
check_alpha_mle_normality(
    alpha=3.0,
    x_m=2.0,
    sample_sizes=[100, 200, 500, 1000],
    n_sim=3000
)


def inspect_xm_mle_distribution(alpha, x_m, n, n_sim=3000):
    # look at x_m hat separately
    alpha_hats, xm_hats = simulate_mle(alpha, x_m, n, n_sim=n_sim)

    print(f"\n=== Distribution of x_m_hat_MLE for n={n} ===")
    print(f"True x_m = {x_m}")
    print(f"Mean(x_m_hat_MLE) = {np.mean(xm_hats):.6f}")
    print(f"Std(x_m_hat_MLE)  = {np.std(xm_hats, ddof=1):.6f}")

    plt.figure(figsize=(8, 4))
    plt.hist(xm_hats, bins=50, density=True)
    plt.axvline(x_m, linestyle='--', linewidth=2)
    plt.title(f"Histogram of x_m_hat_MLE = X_(1), n={n}")
    plt.xlabel("x_m_hat_MLE")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    stats.probplot(xm_hats, dist="norm", plot=plt)
    plt.title(f"QQ-plot of x_m_hat_MLE, n={n}")
    plt.grid(True)
    plt.show()


# try a few sample sizes
inspect_xm_mle_distribution(alpha=3.0, x_m=2.0, n=100, n_sim=3000)
inspect_xm_mle_distribution(alpha=3.0, x_m=2.0, n=500, n_sim=3000)
inspect_xm_mle_distribution(alpha=3.0, x_m=2.0, n=1000, n_sim=3000)

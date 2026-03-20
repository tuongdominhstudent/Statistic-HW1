import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_pareto(alpha, x_m, n):
    # generate Pareto data
    return x_m * (1 + np.random.pareto(alpha, size=n))


def pareto_mom_estimator(sample):
    # MOM estimates
    x_min = np.min(sample)
    x_bar = np.mean(sample)
    alpha_hat = x_bar / (x_bar - x_min)
    return alpha_hat, x_min


def bootstrap_mom_variance(sample, B=2000):
    n = len(sample)

    # save bootstrap estimates
    alpha_boot = np.empty(B)
    xm_boot = np.empty(B)

    for b in range(B):
        boot_sample = np.random.choice(sample, size=n, replace=True)
        alpha_hat_b, xm_hat_b = pareto_mom_estimator(boot_sample)
        alpha_boot[b] = alpha_hat_b
        xm_boot[b] = xm_hat_b

    alpha_var_boot = np.var(alpha_boot, ddof=1)
    xm_var_boot = np.var(xm_boot, ddof=1)

    return alpha_var_boot, xm_var_boot, alpha_boot, xm_boot


def run_bootstrap_experiment(alpha, x_m, sample_sizes, B=2000):
    # keep variances for each n
    alpha_vars = []
    xm_vars = []

    for n in sample_sizes:
        sample = generate_pareto(alpha, x_m, n)

        alpha_hat, xm_hat = pareto_mom_estimator(sample)
        alpha_var_boot, xm_var_boot, alpha_boot, xm_boot = bootstrap_mom_variance(sample, B=B)

        alpha_vars.append(alpha_var_boot)
        xm_vars.append(xm_var_boot)

        print(f"n = {n}")
        print(f"Original alpha_hat_MOM = {alpha_hat:.6f}")
        print(f"Original x_m_hat_MOM   = {xm_hat:.6f}")
        print(f"Bootstrap Var(alpha_hat) = {alpha_var_boot:.6f}")
        print(f"Bootstrap Var(x_m_hat)   = {xm_var_boot:.6f}")
        print("-" * 50)

    return alpha_vars, xm_vars


def plot_bootstrap_variances(sample_sizes, alpha_vars, xm_vars, title_suffix=""):
    # plot alpha variance
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, alpha_vars, marker='o')
    plt.title(f"Bootstrap variance of alpha_hat_MOM vs sample size {title_suffix}")
    plt.xlabel("Sample size n")
    plt.ylabel("Bootstrap variance")
    plt.grid(True)
    plt.show()

    # plot x_m variance
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, xm_vars, marker='o')
    plt.title(f"Bootstrap variance of x_m_hat_MOM vs sample size {title_suffix}")
    plt.xlabel("Sample size n")
    plt.ylabel("Bootstrap variance")
    plt.grid(True)
    plt.show()


sample_sizes = [100, 200, 500, 1000, 2000]

# case where variance exists
print("CASE 1: alpha = 3.0, x_m = 2.0")
alpha_vars_1, xm_vars_1 = run_bootstrap_experiment(
    alpha=3.0,
    x_m=2.0,
    sample_sizes=sample_sizes,
    B=2000
)

plot_bootstrap_variances(sample_sizes, alpha_vars_1, xm_vars_1, title_suffix="(alpha=3.0, x_m=2.0)")


# case with heavier tail
print("\nCASE 2: alpha = 1.5, x_m = 2.0")
alpha_vars_2, xm_vars_2 = run_bootstrap_experiment(
    alpha=1.5,
    x_m=2.0,
    sample_sizes=sample_sizes,
    B=2000
)

plot_bootstrap_variances(sample_sizes, alpha_vars_2, xm_vars_2, title_suffix="(alpha=1.5, x_m=2.0)")

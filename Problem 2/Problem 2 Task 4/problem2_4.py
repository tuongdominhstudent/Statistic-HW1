import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_pareto(alpha, x_m, n):
    # generate Pareto data
    return x_m * (1 + np.random.pareto(alpha, size=n))


def pareto_mom_estimator(sample):
    # MOM estimate
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
    
    return alpha_var_boot, xm_var_boot

def fisher_variance_alpha(alpha, n):
    # asymptotic Fisher variance
    return alpha**2 / n

def compare_bootstrap_and_fisher(alpha, x_m, sample_sizes, B=2000):
    # save all variances
    alpha_boot_vars = []
    alpha_fisher_vars = []
    xm_boot_vars = []

    for n in sample_sizes:
        sample = generate_pareto(alpha, x_m, n)
        alpha_hat, xm_hat = pareto_mom_estimator(sample)
        alpha_var_boot, xm_var_boot = bootstrap_mom_variance(sample, B=B)
        alpha_var_fisher = fisher_variance_alpha(alpha, n)

        alpha_boot_vars.append(alpha_var_boot)
        alpha_fisher_vars.append(alpha_var_fisher)
        xm_boot_vars.append(xm_var_boot)

        print(f"n = {n}")
        print(f"alpha_hat_MOM = {alpha_hat:.6f}")
        print(f"x_m_hat_MOM   = {xm_hat:.6f}")
        print(f"Bootstrap Var(alpha_hat) = {alpha_var_boot:.6f}")
        print(f"Fisher Var(alpha)        = {alpha_var_fisher:.6f}")
        print(f"Bootstrap Var(x_m_hat)   = {xm_var_boot:.6f}")
        print("-" * 55)

    return alpha_boot_vars, alpha_fisher_vars, xm_boot_vars

sample_sizes = [100, 200, 500, 1000, 2000]

# compare the two variance ideas
alpha_boot_vars, alpha_fisher_vars, xm_boot_vars = compare_bootstrap_and_fisher(
    alpha=3.0,
    x_m=2.0,
    sample_sizes=sample_sizes,
    B=2000
)

plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, alpha_boot_vars, marker='o', label='Bootstrap variance of alpha_hat_MOM')
plt.plot(sample_sizes, alpha_fisher_vars, marker='s', label='Fisher asymptotic variance alpha^2/n')
plt.title("Bootstrap vs Fisher variance for alpha")
plt.xlabel("Sample size n")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)
plt.show()

# also plot variance for x_m
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, xm_boot_vars, marker='o')
plt.title("Bootstrap variance of x_m_hat_MOM")
plt.xlabel("Sample size n")
plt.ylabel("Variance")
plt.grid(True)
plt.show()

import numpy as np

np.random.seed(1)

# 97.5% z value
z_975 = 1.959963984540054

miu = 10
sigma = 2
n = 200
bootstrap_resample = 2000
repeat = 10

def empirical_tao(sample):
    # use the 195th ordered value
    return np.sort(sample)[194]

def MLE_tao(sample):
    # plug in sample mean and sd
    MLE_miu = np.mean(sample)
    MLE_sigma = np.sqrt(np.mean((sample - MLE_miu)**2))
    return MLE_miu + z_975 * MLE_sigma

def bootstrap_SE_empirical(sample, bootstrap_resample = 2000):
    n = len(sample)
    arr_of_bootstrap_est = np.empty(bootstrap_resample)
    for b in range(bootstrap_resample):
        # sample with replacement
        bootstrap_sample = np.random.choice(sample, size = n, replace = True)
        arr_of_bootstrap_est[b] = empirical_tao(bootstrap_sample)
    return np.std(arr_of_bootstrap_est, ddof=1)

tao_true = miu + z_975 * sigma

# formula for MLE standard error
SE_MLE_in_theory = (sigma / np.sqrt(n)) * np.sqrt(1 + z_975**2 / 2)

print(f"True tao = {tao_true:.4f}")
print(f"In theory, SE of MLE = {SE_MLE_in_theory:.4f}")
print()

boot_SE_list = []
for r in range(repeat):
    # generate one normal sample
    sample = np.random.normal(miu, sigma, n)

    tao_empirical = empirical_tao(sample)
    tao_MLE = MLE_tao(sample)
    boot_SE = bootstrap_SE_empirical(sample, bootstrap_resample = bootstrap_resample)
    boot_SE_list.append(boot_SE)

    print(f"Run {r + 1}:")
    print(f"Empirical tao x_(195) = {tao_empirical:.4f}")
    print(f"MLE tao = {tao_MLE:.4f}")
    print(f"Bootstrap SE empirical = {boot_SE:.4f}")
    print()

print(f"Average bootstrap SE(empirical) over {repeat} runs = {np.mean(boot_SE_list):.4f}")
print(f"Theoretical SE(MLE) = {SE_MLE_in_theory:.4f}")

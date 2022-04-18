import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))


def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance


xs = [x / 10.0 for x in range(-50, 50)]
ys = [normal_pdf(x, sigma=1) for x in xs]
# plt.plot(xs, ys, '-', label='mu=0, sigma=1')
# plt.show()


print(f"std {np.std(ys)}")
print(f"variance {variance(ys)}")
print(f"25 percentile {np.percentile(ys, 25)}")
print(f"75 percentile {np.percentile(ys, 75)}")
print(f"diff 75-25 {np.percentile(ys, 75) - np.percentile(ys, 25)}")

range = np.max(ys) - np.min(ys)
print(range)

# zscore = 0.1
# zscores = [(value - mean) / standard_deviation for value in values]

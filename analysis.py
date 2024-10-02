import numpy as np
from scipy.stats import norm


def generate_gaussian_sum_fixed(n, sigma):
    '''
    :param n: how many participants
    :param sigma: scale
    :return: list [c1,c2,c3,……,cn], ci represent how many people want to talk to pi
    '''
    samples = norm.rvs(loc=1, scale=sigma, size=n)
    samples = np.maximum(0, np.round(samples)).astype(int)
    total_sum = np.sum(samples)

    while total_sum != n:
        diff = int(total_sum - n)
        abs_diff = abs(diff)

        if abs_diff > n:
            abs_diff = n

        indices = np.random.choice(n, abs_diff, replace=False)

        if diff > 0:
            samples[indices] = np.maximum(0, samples[indices] - 1)
        else:
            samples[indices] += 1

        total_sum = np.sum(samples)

    return samples.tolist()

def count_values(arr, n):
    counts = [0] * n  # Initialize a list to hold counts for each value
    for value in arr:
        counts[value] += 1  # Increment the count for the value
    return counts

# Example usage:


n = 10

sigma = 4

result = generate_gaussian_sum_fixed(n, sigma)
print(result)
arr = result
n = len(arr)
counts = count_values(arr, n)
print(counts)
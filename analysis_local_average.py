import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from protocol import generate_gaussian_request_count


def count_values(request_count):
    counts = [0] * len(request_count)  # Initialize a list to hold counts for each value
    for value in request_count:
        counts[value] += 1  # Increment the count for the value
    return counts


def calculate_average_difference(n, sigma, epsilon, runs=100):
    total_original = 0
    total_now = 0

    for _ in range(runs):
        request_count = generate_gaussian_request_count(n, sigma)
        request_count_count = count_values(request_count)

        sum_original = 0
        for i in range(1, n):
            sum_original += 1 * (request_count_count[i])
        average_original = sum_original / n
        total_original += average_original

        sum_now = 0
        for i in range(1, n):
            sum_now += (math.exp(epsilon) / (n - 1 + math.exp(epsilon))) * (request_count_count[i])
        average_now = sum_now / n
        total_now += average_now

    # Take the average over 50 runs
    return total_original / runs, total_now / runs


def compare_epsilon_effect_for_average(n, sigma, epsilon_values):
    original_avgs = []
    noisy_avgs = []

    for epsilon in epsilon_values:
        avg_original, avg_now = calculate_average_difference(n, sigma, epsilon)
        original_avgs.append(avg_original)
        noisy_avgs.append(avg_now)

    # Smooth the data
    epsilon_smooth = np.linspace(min(epsilon_values), max(epsilon_values), 300)
    smooth_original_avgs = make_interp_spline(epsilon_values, original_avgs)(epsilon_smooth)
    smooth_noisy_avgs = make_interp_spline(epsilon_values, noisy_avgs)(epsilon_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_smooth, smooth_original_avgs, label="Original Average", color='blue')
    plt.plot(epsilon_smooth, smooth_noisy_avgs, label="Noisy Average", color='red')

    plt.title(f"Effect of Epsilon on Average (n={n}, sigma={sigma})")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Difference")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_n_effect_for_average(sigma, epsilon, n_values, runs=50):
    original_avgs = []
    noisy_avgs = []

    for n in n_values:
        avg_original, avg_now = calculate_average_difference(n, sigma, epsilon, runs)
        original_avgs.append(avg_original)
        noisy_avgs.append(avg_now)

    # Smooth the data
    n_smooth = np.linspace(min(n_values), max(n_values), 300)
    smooth_original_avgs = make_interp_spline(n_values, original_avgs)(n_smooth)
    smooth_noisy_avgs = make_interp_spline(n_values, noisy_avgs)(n_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(n_smooth, smooth_original_avgs, label="Original Average", color='blue')
    plt.plot(n_smooth, smooth_noisy_avgs, label="Noisy Average", color='red')

    plt.title(f"Effect of n on Average (sigma={sigma}, epsilon={epsilon})")
    plt.xlabel("Number of Participants (n)")
    plt.ylabel("Average Difference")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_sigma_effect_for_average(n, epsilon, sigma_values, runs=50):
    original_avgs = []
    noisy_avgs = []

    for sigma in sigma_values:
        avg_original, avg_now = calculate_average_difference(n, sigma, epsilon, runs)
        original_avgs.append(avg_original)
        noisy_avgs.append(avg_now)

    # Smooth the data
    sigma_smooth = np.linspace(min(sigma_values), max(sigma_values), 300)
    smooth_original_avgs = make_interp_spline(sigma_values, original_avgs)(sigma_smooth)
    smooth_noisy_avgs = make_interp_spline(sigma_values, noisy_avgs)(sigma_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_smooth, smooth_original_avgs, label="Original Average", color='blue')
    plt.plot(sigma_smooth, smooth_noisy_avgs, label="Noisy Average", color='red')

    plt.title(f"Effect of Sigma on Average (n={n}, epsilon={epsilon})")
    plt.xlabel("Sigma")
    plt.ylabel("Average Difference")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example: Compare the effect of different sigma values (with smoothing)
sigma_values = np.arange(1, 20, 2)
compare_sigma_effect_for_average(n=100, epsilon=8, sigma_values=sigma_values)
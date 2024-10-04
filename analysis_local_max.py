import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import make_interp_spline


def calculate_max_difference(x, n, epsilon):
    probability_original = 1 / x
    probability_now = (math.exp(epsilon) / (n - 1 + math.exp(epsilon))) * (1 / x)
    return probability_original, probability_now


def compare_epsilon_effect(x, n, epsilon_values):
    original_probs = []
    noisy_probs = []

    for epsilon in epsilon_values:
        prob_original, prob_now = calculate_max_difference(x, n, epsilon)
        original_probs.append(prob_original)
        noisy_probs.append(prob_now)

    # Smooth the data
    epsilon_smooth = np.linspace(min(epsilon_values), max(epsilon_values), 300)
    smooth_original_probs = make_interp_spline(epsilon_values, original_probs)(epsilon_smooth)
    smooth_noisy_probs = make_interp_spline(epsilon_values, noisy_probs)(epsilon_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_smooth, smooth_original_probs, label="Original Probability", color='blue')
    plt.plot(epsilon_smooth, smooth_noisy_probs, label="Noisy Probability", color='red')

    plt.title(f"Effect of Epsilon on Probability (x={x}, n={n})")
    plt.xlabel("Epsilon")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_n_effect(x, epsilon, n_values):
    original_probs = []
    noisy_probs = []

    for n in n_values:
        prob_original, prob_now = calculate_max_difference(x, n, epsilon)
        original_probs.append(prob_original)
        noisy_probs.append(prob_now)

    # Smooth the data
    n_smooth = np.linspace(min(n_values), max(n_values), 300)
    smooth_original_probs = make_interp_spline(n_values, original_probs)(n_smooth)
    smooth_noisy_probs = make_interp_spline(n_values, noisy_probs)(n_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(n_smooth, smooth_original_probs, label="Original Probability", color='blue')
    plt.plot(n_smooth, smooth_noisy_probs, label="Noisy Probability", color='red')

    plt.title(f"Effect of n on Probability (x={x}, epsilon={epsilon})")
    plt.xlabel("Number of Participants (n)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_x_effect(n, epsilon, x_values):
    original_probs = []
    noisy_probs = []

    for x in x_values:
        prob_original, prob_now = calculate_max_difference(x, n, epsilon)
        original_probs.append(prob_original)
        noisy_probs.append(prob_now)

    # Smooth the data
    x_smooth = np.linspace(min(x_values), max(x_values), 300)
    smooth_original_probs = make_interp_spline(x_values, original_probs)(x_smooth)
    smooth_noisy_probs = make_interp_spline(x_values, noisy_probs)(x_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, smooth_original_probs, label="Original Probability", color='blue')
    plt.plot(x_smooth, smooth_noisy_probs, label="Noisy Probability", color='red')

    plt.title(f"Effect of x on Probability (n={n}, epsilon={epsilon})")
    plt.xlabel("x (Number of Parties with Same Target)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example: Comparing the effect of different x values (with smoothing)
x_values = np.arange(1, 70, 5)  # Generate x values from 1 to 70 with a step of 5
compare_x_effect(n=100, epsilon=5, x_values=x_values)
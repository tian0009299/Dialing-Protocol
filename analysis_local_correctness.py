import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from protocol import generate_gaussian_request_count, generate_inputs, generate_outputs
from DP import local_dp


def calculate_correctness(n, sigma, epsilon):
    request_count = generate_gaussian_request_count(n, sigma)
    inputs = generate_inputs(request_count)
    outputs = generate_outputs(inputs=inputs, request_count=request_count)
    inputs_with_noise, request_count_with_noise = local_dp(inputs, epsilon=epsilon)
    outputs_with_noise = generate_outputs(inputs=inputs_with_noise, request_count=request_count_with_noise)
    # print("Request_count: ", request_count)
    # print("Inputs:", inputs)
    # print("Outputs:", outputs)
    # print("Inputs_with_noise:", inputs_with_noise)
    # print("Outputs_with_noise:", outputs_with_noise)
    correctness = sum(1 for a, b in zip(inputs, outputs) if a == b)
    correctness_with_noise = sum(1 for a, b in zip(inputs, outputs_with_noise) if a == b)
    # print("Correctness: ", correctness)
    # print("Correctness_with_noise: ", correctness_with_noise)
    return correctness, correctness_with_noise


def analyze_correctness(m, n, sigma, epsilon):
    # Lists to store correctness and correctness_with_noise
    correctness = []
    correctness_with_noise = []

    # Run the test_correctness multiple times and output the results directly
    for run in range(1, m + 1):
        corr, corr_with_noise = calculate_correctness(n, sigma, epsilon)
        correctness.append(corr)
        correctness_with_noise.append(corr_with_noise)
        # print(f"Run {run}: Correctness = {corr}, Correctness with Noise = {corr_with_noise}")

    # Calculate the average correctness and correctness_with_noise
    avg_correctness = np.mean(correctness)
    avg_correctness_with_noise = np.mean(correctness_with_noise)

    # Plot the graph
    plt.figure(figsize=(10, 6))

    # Plot the curves for correctness and correctness_with_noise
    plt.plot(correctness, label="Correctness (No Noise)", color='blue', marker='o')
    plt.plot(correctness_with_noise, label="Correctness (With Noise)", color='red', marker='x')

    # Plot horizontal lines for the averages
    plt.axhline(avg_correctness, color='blue', linestyle='--', label="Average Correctness (No Noise)")
    plt.axhline(avg_correctness_with_noise, color='red', linestyle='--', label="Average Correctness (With Noise)")

    # Set titles and labels
    plt.title(f"Correctness Comparison (n={n}, sigma={sigma}, epsilon={epsilon})")
    plt.xlabel("Run")
    plt.ylabel("Correctness")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


def compare_n_effect_on_correctness(sigma, epsilon, n_values, runs=50):
    avg_correctness = []
    avg_correctness_with_noise = []

    for n in n_values:
        total_correctness = 0
        total_correctness_with_noise = 0

        # Run 'runs' times and take the average
        for _ in range(runs):
            correctness, correctness_with_noise = calculate_correctness(n, sigma, epsilon)
            total_correctness += correctness
            total_correctness_with_noise += correctness_with_noise

        avg_correctness.append(total_correctness / runs)  # Correctness divided by the number of runs
        avg_correctness_with_noise.append(total_correctness_with_noise / runs)

    # Smooth the data
    n_smooth = np.linspace(min(n_values), max(n_values), 300)
    smooth_correctness = make_interp_spline(n_values, avg_correctness)(n_smooth)
    smooth_correctness_with_noise = make_interp_spline(n_values, avg_correctness_with_noise)(n_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(n_smooth, smooth_correctness, label="Correctness (No Noise)", color='blue')
    plt.plot(n_smooth, smooth_correctness_with_noise, label="Correctness (With Noise)", color='red')

    plt.title(f"Effect of n on Correctness (sigma={sigma}, epsilon={epsilon})")
    plt.xlabel("Number of Participants (n)")
    plt.ylabel("Correctness (Average)")
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_correctness_sigma_effect(n, epsilon, sigma_values, runs=50):
    avg_correctness = []
    avg_correctness_with_noise = []

    for sigma in sigma_values:
        total_correctness = 0
        total_correctness_with_noise = 0

        # Run test_correctness multiple times and calculate the average
        for _ in range(runs):
            correctness, correctness_with_noise = calculate_correctness(n, sigma, epsilon)
            total_correctness += correctness
            total_correctness_with_noise += correctness_with_noise

        avg_correctness.append(total_correctness / runs)
        avg_correctness_with_noise.append(total_correctness_with_noise / runs)

    # Smooth the data
    sigma_smooth = np.linspace(min(sigma_values), max(sigma_values), 300)
    smooth_correctness = make_interp_spline(sigma_values, avg_correctness)(sigma_smooth)
    smooth_correctness_with_noise = make_interp_spline(sigma_values, avg_correctness_with_noise)(sigma_smooth)

    # Plot the graph
    plt.figure(figsize=(10, 6))

    plt.plot(sigma_smooth, smooth_correctness, label="Correctness (No Noise)", color='blue')
    plt.plot(sigma_smooth, smooth_correctness_with_noise, label="Correctness (With Noise)", color='red')

    plt.title(f"Effect of Sigma on Correctness (n={n}, epsilon={epsilon})")
    plt.xlabel("Sigma")
    plt.ylabel("Average Correctness")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


# Example: Compare the effect of different n values on correctness
n_values = np.arange(50, 1000, 50)  # Generate n values from 50 to 1000 with a step of 50
compare_n_effect_on_correctness(sigma=8, epsilon=6, n_values=n_values)
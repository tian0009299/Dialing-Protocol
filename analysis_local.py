import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from protocol import generate_gaussian_request_count, generate_inputs, generate_outputs
from DP import local_dp, exponential_mechanism_simplified_adjusted


def calculate_max_difference(x, n, epsilon):
    probability_original = 1 / x
    probability_now = (math.exp(epsilon) / (n - 1 + math.exp(epsilon))) * (1/x)
    return probability_original, probability_now


def count_values(request_count):
    counts = [0] * len(request_count)  # Initialize a list to hold counts for each value
    for value in request_count:
        counts[value] += 1  # Increment the count for the value
    return counts


def calculate_average_difference(n, sigma, epsilon):
    request_count = generate_gaussian_request_count(n, sigma)
    request_count_count = count_values(request_count)
    # print("request_count: ", request_count)
    # print("request_count_count: ", request_count_count)
    sum_original = 0
    for i in range(1, n):
        sum_original += 1 * (request_count_count[i])
    average_original = sum_original / n
    # print("average_original: ", average_original)
    sum_now = 0
    for i in range(1, n):
        sum_now += (math.exp(epsilon) / (n - 1 + math.exp(epsilon))) * (request_count_count[i])
    average_now = sum_now / n
    # print("average_now: ", average_now)
    return average_original, average_now


def calculate_correctness(n, sigma, epsilon):
    request_count = generate_gaussian_request_count(n, sigma)
    inputs = generate_inputs(request_count)
    outputs = generate_outputs(inputs=inputs, request_count=request_count)
    inputs_with_noise, request_count_with_noise = local_dp(inputs, epsilon=epsilon)
    outputs_with_noise = generate_outputs(inputs=inputs_with_noise, request_count=request_count_with_noise)
    # print("Request_count: ", request_count)
    # print("Inputs:", inputs)
    # print("Outputs:", outputs)
    # print("Inputs_fake:", inputs_with_noise)
    # print("Outputs_fake:", outputs_with_noise)
    correctness = sum(1 for a, b in zip(inputs, outputs) if a == b)
    correctness_with_noise = sum(1 for a, b in zip(inputs, outputs_with_noise) if a == b)
    # print("Correctness: ", correctness)
    # print("Correctness_fake: ", correctness_with_noise)
    return correctness, correctness_with_noise


def analyze_correctness(m, n, sigma, epsilon):
    # 用于存储 correctness 和 correctness_with_noise 的列表
    correctness = []
    correctness_with_noise = []

    # 多次运行 test_correctness 并直接输出结果
    for run in range(1, m + 1):
        corr, corr_with_noise = calculate_correctness(n, sigma, epsilon)
        correctness.append(corr)
        correctness_with_noise.append(corr_with_noise)
        # print(f"Run {run}: Correctness = {corr}, Correctness with Noise = {corr_with_noise}")

    # 计算 correctness 和 correctness_with_noise 的平均值
    avg_correctness = np.mean(correctness)
    avg_correctness_with_noise = np.mean(correctness_with_noise)

    # 绘制图表
    plt.figure(figsize=(10, 6))

    # 绘制 correctness 和 correctness_with_noise 的曲线
    plt.plot(correctness, label="Correctness (No Noise)", color='blue', marker='o')
    plt.plot(correctness_with_noise, label="Correctness (With Noise)", color='red', marker='x')

    # 绘制平均值的直线
    plt.axhline(avg_correctness, color='blue', linestyle='--', label="Average Correctness (No Noise)")
    plt.axhline(avg_correctness_with_noise, color='red', linestyle='--', label="Average Correctness (With Noise)")

    # 图表标题和标签
    plt.title(f"Correctness Comparison (n={n}, sigma={sigma}, epsilon={epsilon})")
    plt.xlabel("Run")
    plt.ylabel("Correctness")
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()


def analyze_correctness_epsilon_effect(n, sigma, epsilon_values, runs=50):
    avg_correctness = []
    avg_correctness_with_noise = []

    for epsilon in epsilon_values:
        total_correctness = 0
        total_correctness_with_noise = 0

        # 多次运行 test_correctness 并计算平均值
        for _ in range(runs):
            correctness, correctness_with_noise = calculate_correctness(n, sigma, epsilon)
            total_correctness += correctness
            total_correctness_with_noise += correctness_with_noise

        avg_correctness.append(total_correctness / runs)
        avg_correctness_with_noise.append(total_correctness_with_noise / runs)

    # 平滑处理
    epsilon_smooth = np.linspace(min(epsilon_values), max(epsilon_values), 300)
    smooth_correctness = make_interp_spline(epsilon_values, avg_correctness)(epsilon_smooth)
    smooth_correctness_with_noise = make_interp_spline(epsilon_values, avg_correctness_with_noise)(epsilon_smooth)

    # 绘制图表
    plt.figure(figsize=(10, 6))

    plt.plot(epsilon_smooth, smooth_correctness, label="Correctness (No Noise)", color='blue')
    plt.plot(epsilon_smooth, smooth_correctness_with_noise, label="Correctness (With Noise)", color='red')

    plt.title(f"Effect of Epsilon on Correctness (n={n}, sigma={sigma})")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Correctness")
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()


def analyze_correctness_sigma_effect(n, epsilon, sigma_values, runs=50):
    avg_correctness = []
    avg_correctness_with_noise = []

    for sigma in sigma_values:
        total_correctness = 0
        total_correctness_with_noise = 0

        # 多次运行 test_correctness 并计算平均值
        for _ in range(runs):
            correctness, correctness_with_noise = calculate_correctness(n, sigma, epsilon)
            total_correctness += correctness
            total_correctness_with_noise += correctness_with_noise

        avg_correctness.append(total_correctness / runs)
        avg_correctness_with_noise.append(total_correctness_with_noise / runs)

    # 平滑处理
    sigma_smooth = np.linspace(min(sigma_values), max(sigma_values), 300)
    smooth_correctness = make_interp_spline(sigma_values, avg_correctness)(sigma_smooth)
    smooth_correctness_with_noise = make_interp_spline(sigma_values, avg_correctness_with_noise)(sigma_smooth)

    # 绘制图表
    plt.figure(figsize=(10, 6))

    plt.plot(sigma_smooth, smooth_correctness, label="Correctness (No Noise)", color='blue')
    plt.plot(sigma_smooth, smooth_correctness_with_noise, label="Correctness (With Noise)", color='red')

    plt.title(f"Effect of Sigma on Correctness (n={n}, epsilon={epsilon})")
    plt.xlabel("Sigma")
    plt.ylabel("Average Correctness")
    plt.legend()
    plt.grid(True)

    # 显示图表
    plt.show()

analyze_correctness(m=100, n=100, sigma=5, epsilon=np.log(50))
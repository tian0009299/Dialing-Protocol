import numpy as np
from scipy.stats import norm
import math


def generate_gaussian_request_count(n, sigma):
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


def generate_inputs(request_count):
    n = len(request_count)
    inputs = [-1] * n
    available_receivers = []

    for i in range(n):
        available_receivers.extend([i+1] * request_count[i])

    np.random.shuffle(available_receivers)

    for i in range(n):
        inputs[i] = available_receivers[i]

    return inputs


def local_dp(inputs, epsilon):
    n = len(inputs)
    p_true = np.exp(epsilon) / (n - 1 + np.exp(epsilon))
    p_fake = (n - 1) / (n - 1 + np.exp(epsilon))
    inputs_with_noise = inputs.copy()

    for i in range(n):
        if np.random.rand() < p_fake:
            possible_choices = [x for x in range(1, n + 1) if x != inputs[i]]
            inputs_with_noise[i] = np.random.choice(possible_choices)

    request_count_with_noise = [0] * n
    for request in inputs_with_noise:
        request_count_with_noise[request - 1] += 1  # 统计每个P_i收到的请求数量

    return inputs_with_noise, request_count_with_noise




# Example usage:
# request_count = [4, 0, 0, 0, 1, 0, 0, 0, 0, 5]
# inputs = generate_inputs(request_count)
# inputs_with_noise, request_count_with_noise = local_dp(inputs, np.log(10))
# print(inputs, request_count)
# print(inputs_with_noise, request_count_with_noise)


def generate_outputs(inputs, request_count):
    n = len(inputs)

    # Step 1: 找到 request_count 中值为 0 的项，并使用集合 NO
    NO = {i + 1 for i, count in enumerate(request_count) if count == 0}  # 使用集合存储 NO
    outputs = inputs.copy()  # 初始化 outputs 为 inputs 的副本
    processed_indices = []  # 用列表存储已经处理过的索引

    # Step 2: 遍历 request_count 中大于 0 的项
    for idx, count in enumerate(request_count):
        if count == 0:
            continue  # 如果为0，跳过处理


        # 当前处理的是 (idx + 1) 对应的值
        target_value = idx + 1

        # 找到 inputs 中所有等于 target_value 的项的索引，且这些索引还没有被处理过
        target_indices = [i for i, value in enumerate(inputs) if value == target_value]


        # 从这些 target_indices 中随机选择一个保留
        keep_index = np.random.choice(target_indices)


        # 其余项进行替换
        for i in target_indices:
            if i == keep_index:
                continue  # 保留该项，不做替换
            if NO:
                new_value = NO.pop()  # 从 NO 集合中随机选择并移除一个值
                outputs[i] = new_value  # 替换当前值

    return outputs



def test_correctness(n, sigma, epsilon):
    request_count = generate_gaussian_request_count(n, sigma)
    inputs = generate_inputs(request_count)
    outputs = generate_outputs(inputs=inputs, request_count=request_count)
    inputs_with_noise, request_count_with_noise = local_dp(inputs, epsilon=epsilon)
    outputs_with_noise = generate_outputs(inputs=inputs_with_noise, request_count=request_count_with_noise)
    print("Request_count: ", request_count)
    print("Inputs:", inputs)
    print("Outputs:", outputs)
    print("Inputs_fake:", inputs_with_noise)
    print("Outputs_fake:", outputs_with_noise)
    correctness = sum(1 for a, b in zip(inputs, outputs) if a == b)
    correctness_with_noise = sum(1 for a, b in zip(inputs, outputs_with_noise) if a == b)
    print("Correctness: ", correctness)
    print("Correctness_fake: ", correctness_with_noise)


def calculate_max_difference(x, n, epsilon):
    probability_original = 1 / x
    probability_now = (math.exp(epsilon) / (n - 1 + math.exp(epsilon))) * (1/x)

    return probability_original, probability_now




n = 1000
sigma = 500
epsilon = np.log(50)
test_correctness(n, sigma, epsilon)

# x = 1
# n = 1000
# epsilon = np.log(50)
# probability_original, probability_now = calculate_max_difference(x, n, epsilon)
# print("original: ", probability_original)
# print("now: ", probability_now)


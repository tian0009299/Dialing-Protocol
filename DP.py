import numpy as np
import math
import random


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


def exponential_mechanism_simplified(output, n, epsilon):
    # 1. 遍历不同的 u(x, r) 的值，从 n 到 1，计算权重
    def calculate_weights(n, epsilon):
        weights = {}
        weights[0] = np.exp(epsilon * n / 4)
        for i in range(2, n + 1):  # 从2到n（不考虑1项变化）
            weight = np.exp((epsilon * (n - i)) / 4)  # 使用你的公式
            weights[i] = weight
        return weights

    # 2. 计算sum latex中的公式
    def calculate_sum(weights):
        sum_weight = 0
        sum_weight += math.perm(n, 0) * weights[0]
        for i in range(2, n + 1):
            sum_weight += (math.perm(n, i) * (math.factorial(i)-1)/math.factorial(i)) * weights[i]
        return sum_weight

    # 3. 计算改变不同个数元素的概率
    def calculate_probabilities(weights, sum_weights):
        probabilities = {}
        for i in weights:
            if i == 0:
                # 针对 i = 0 的特殊操作
                probabilities[i] = math.perm(n, 0) * weights[0] / sum_weights
            elif i == 1:
                pass
            else:
                # 其他情况按原来的方式处理
                probabilities[i] = (math.perm(n, i) * (math.factorial(i)-1)/math.factorial(i)) * weights[i] / sum_weights

        return probabilities

    # 4. 根据概率随机采样选择要改变几个元素
    def select_change_size(probabilities):
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        return np.random.choice(keys, p=values)

    # 5. 随机选择要改变的 x 个元素，并打乱顺序，确保不是原来的顺序
    def shuffle_elements(output, change_size):
        indices_to_change = random.sample(range(n), change_size)  # 随机选择 x 个元素的索引
        indices_to_change = sorted(indices_to_change)
        print("indices_to_change: ", indices_to_change)
        elements_to_change = [output[i] for i in indices_to_change]  # 获取这些元素

        random.shuffle(elements_to_change)  # 打乱元素顺序

        # 生成新的output
        new_output = output.copy()
        for idx, new_element in zip(indices_to_change, elements_to_change):
            new_output[idx] = new_element
        return new_output

    # 1. 计算权重
    weights = calculate_weights(n, epsilon)
    print(weights)

    # 2. 计算 sum
    sum_weights = calculate_sum(weights)
    print(sum_weights)

    # 3. 计算概率
    probabilities = calculate_probabilities(weights, sum_weights)
    print(probabilities)

    # 4. 随机选择要改变的元素个数
    change_size = select_change_size(probabilities)
    print(change_size)

    # 5. 随机选择 x 个元素并打乱
    perturbed_output = shuffle_elements(output, change_size)
    print(perturbed_output)

    # 6. 输出扰动后的 output
    return perturbed_output


def exponential_mechanism_simplified_adjusted(output, n, epsilon):
    # 1. 遍历不同的 u(x, r) 的值，从 n 到 1，计算权重
    def calculate_weights(n, epsilon):
        weights = {}
        weights[0] = np.exp(epsilon * n / 4)
        for i in range(2, n + 1):  # 从2到n（不考虑1项变化）
            weight = np.exp((epsilon * (n - i)) / 4)  # 使用你的公式
            weights[i] = weight
        return weights

    # 2. 计算sum latex中的公式
    def calculate_sum(weights):
        sum_weight = 0
        sum_weight += weights[0]
        for i in range(2, n + 1):
            sum_weight += weights[i]
        return sum_weight

    # 3. 计算改变不同个数元素的概率
    def calculate_probabilities(weights, sum_weights):
        probabilities = {}
        for i in weights:
            if i == 0:
                # 针对 i = 0 的特殊操作
                probabilities[i] = weights[0] / sum_weights
            elif i == 1:
                pass
            else:
                # 其他情况按原来的方式处理
                probabilities[i] = weights[i] / sum_weights

        return probabilities

    # 4. 根据概率随机采样选择要改变几个元素
    def select_change_size(probabilities):
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        return np.random.choice(keys, p=values)

    # 5. 随机选择要改变的 x 个元素，并打乱顺序，确保不是原来的顺序
    def shuffle_elements(output, change_size):
        indices_to_change = random.sample(range(n), change_size)  # 随机选择 x 个元素的索引
        indices_to_change = sorted(indices_to_change)
        print("indices_to_change: ", indices_to_change)
        if len(indices_to_change) > 1:
            elements_to_change = [output[i] for i in indices_to_change]  # 获取这些元素
            shuffled_lst = elements_to_change[:]
            while True:
                random.shuffle(shuffled_lst)  # 打乱列表
                if shuffled_lst != elements_to_change:  # 确保打乱后的列表和原始列表不同
                    break
            new_output = output.copy()
            for idx, new_element in zip(indices_to_change, shuffled_lst):
                new_output[idx] = new_element
            return new_output
        return output




    # 1. 计算权重
    weights = calculate_weights(n, epsilon)
    print(weights)

    # 2. 计算 sum
    sum_weights = calculate_sum(weights)
    print(sum_weights)

    # 3. 计算概率
    probabilities = calculate_probabilities(weights, sum_weights)
    print(probabilities)

    # 4. 随机选择要改变的元素个数
    change_size = select_change_size(probabilities)
    print(change_size)

    # 5. 随机选择 x 个元素并打乱
    perturbed_output = shuffle_elements(output, change_size)
    print(perturbed_output)

    # 6. 输出扰动后的 output
    return perturbed_output


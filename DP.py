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
        request_count_with_noise[request - 1] += 1  # Count the number of requests each P_i receives

    return inputs_with_noise, request_count_with_noise


def exponential_mechanism_simplified(output, n, epsilon):
    # 1. Iterate over different u(x, r) values from n to 1 and calculate weights
    def calculate_weights(n, epsilon):
        weights = {}
        weights[0] = np.exp(epsilon * n / 4)
        for i in range(2, n + 1):  # From 2 to n (ignoring 1)
            weight = np.exp((epsilon * (n - i)) / 4)  # Using your formula
            weights[i] = weight
        return weights

    # 2. Calculate the sum based on the formula in LaTeX
    def calculate_sum(weights):
        sum_weight = 0
        sum_weight += math.perm(n, 0) * weights[0]
        for i in range(2, n + 1):
            sum_weight += (math.perm(n, i) * (math.factorial(i) - 1) / math.factorial(i)) * weights[i]
        return sum_weight

    # 3. Calculate the probability of changing a different number of elements
    def calculate_probabilities(weights, sum_weights):
        probabilities = {}
        for i in weights:
            if i == 0:
                # Special case for i = 0
                probabilities[i] = math.perm(n, 0) * weights[0] / sum_weights
            elif i == 1:
                pass
            else:
                # Handle other cases as before
                probabilities[i] = (math.perm(n, i) * (math.factorial(i) - 1) / math.factorial(i)) * weights[i] / sum_weights

        return probabilities

    # 4. Randomly sample how many elements to change based on the probabilities
    def select_change_size(probabilities):
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        return np.random.choice(keys, p=values)

    # 5. Randomly select x elements to change and shuffle their order to ensure they are not in the original order
    def shuffle_elements(output, change_size):
        indices_to_change = random.sample(range(n), change_size)  # Randomly select indices for x elements
        indices_to_change = sorted(indices_to_change)
        print("indices_to_change: ", indices_to_change)
        elements_to_change = [output[i] for i in indices_to_change]  # Get the elements

        random.shuffle(elements_to_change)  # Shuffle the elements

        # Generate the new output
        new_output = output.copy()
        for idx, new_element in zip(indices_to_change, elements_to_change):
            new_output[idx] = new_element
        return new_output

    # 1. Calculate the weights
    weights = calculate_weights(n, epsilon)
    print(weights)

    # 2. Calculate the sum
    sum_weights = calculate_sum(weights)
    print(sum_weights)

    # 3. Calculate the probabilities
    probabilities = calculate_probabilities(weights, sum_weights)
    print(probabilities)

    # 4. Randomly select how many elements to change
    change_size = select_change_size(probabilities)
    print(change_size)

    # 5. Randomly select x elements and shuffle them
    perturbed_output = shuffle_elements(output, change_size)
    print(perturbed_output)

    # 6. Return the perturbed output
    return perturbed_output


def exponential_mechanism_simplified_adjusted(output, n, epsilon):
    # 1. Iterate over different u(x, r) values from n to 1 and calculate weights
    def calculate_weights(n, epsilon):
        weights = {}
        weights[0] = np.exp(epsilon * n / 4)
        for i in range(2, n + 1):  # From 2 to n (ignoring 1)
            weight = np.exp((epsilon * (n - i)) / 4)  # Using your formula
            weights[i] = weight
        return weights

    # 2. Calculate the sum based on the formula in LaTeX
    def calculate_sum(weights):
        sum_weight = 0
        sum_weight += weights[0]
        for i in range(2, n + 1):
            sum_weight += weights[i]
        return sum_weight

    # 3. Calculate the probability of changing a different number of elements
    def calculate_probabilities(weights, sum_weights):
        probabilities = {}
        for i in weights:
            if i == 0:
                # Special case for i = 0
                probabilities[i] = weights[0] / sum_weights
            elif i == 1:
                pass
            else:
                # Handle other cases as before
                probabilities[i] = weights[i] / sum_weights

        return probabilities

    # 4. Randomly sample how many elements to change based on the probabilities
    def select_change_size(probabilities):
        keys = list(probabilities.keys())
        values = list(probabilities.values())
        return np.random.choice(keys, p=values)

    # 5. Randomly select x elements to change and shuffle their order to ensure they are not in the original order
    def shuffle_elements(output, change_size):
        indices_to_change = random.sample(range(n), change_size)  # Randomly select indices for x elements
        indices_to_change = sorted(indices_to_change)
        print("indices_to_change: ", indices_to_change)
        if len(indices_to_change) > 1:
            elements_to_change = [output[i] for i in indices_to_change]  # Get the elements
            shuffled_lst = elements_to_change[:]
            while True:
                random.shuffle(shuffled_lst)  # Shuffle the list
                if shuffled_lst != elements_to_change:  # Ensure the shuffled list is different from the original
                    break
            new_output = output.copy()
            for idx, new_element in zip(indices_to_change, shuffled_lst):
                new_output[idx] = new_element
            return new_output
        return output

    # 1. Calculate the weights
    weights = calculate_weights(n, epsilon)
    print(weights)

    # 2. Calculate the sum
    sum_weights = calculate_sum(weights)
    print(sum_weights)

    # 3. Calculate the probabilities
    probabilities = calculate_probabilities(weights, sum_weights)
    print(probabilities)

    # 4. Randomly select how many elements to change
    change_size = select_change_size(probabilities)
    print(change_size)

    # 5. Randomly select x elements and shuffle them
    perturbed_output = shuffle_elements(output, change_size)
    print(perturbed_output)

    # 6. Return the perturbed output
    return perturbed_output


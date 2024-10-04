import numpy as np
from scipy.stats import norm


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


def generate_outputs(inputs, request_count):
    n = len(inputs)

    # Step 1: Find the invitees who have no inviter and store them in the set NO
    NO = {i + 1 for i, count in enumerate(request_count) if count == 0}  # Store invitees with no inviter in the set NO
    outputs = inputs.copy()  # Initialize outputs as a copy of inputs
    processed_indices = []  # Store the processed indices in a list

    # Step 2: Iterate over items in request_count that are greater than 0
    for idx, count in enumerate(request_count):
        if count == 0:
            continue  # Skip if count is 0

        # Currently processing the value corresponding to (idx + 1)
        target_value = idx + 1

        # Find all indices in inputs that are equal to target_value and have not been processed
        target_indices = [i for i, value in enumerate(inputs) if value == target_value]

        # Randomly select one index to keep from target_indices
        keep_index = np.random.choice(target_indices)

        # Replace the remaining items
        for i in target_indices:
            if i == keep_index:
                continue  # Keep this item, do not replace it
            if NO:
                new_value = NO.pop()  # Randomly select and remove a value from the set NO
                outputs[i] = new_value  # Replace the current value

    return outputs

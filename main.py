import math
import itertools


def calculate_target_random(input, corrupted):
    # target_dict: a dictionary storing the permutation from the corrupted inviters to their invitees
    # random: store the parties who don't have the inviters (will be randomly assigned by the bulletin board to
    # some inviter)
    # target_re: a list storing the permutation from invitees to inviters
    target_re = [[] for _ in range(len(input))]
    for i in range(len(input)):
        target_re[input[i]-1].append(i+1)
    target_dict = {}
    random = list(range(1, len(input)+1))
    for i in range(len(corrupted)):
        target_dict[corrupted[i]] = input[corrupted[i]-1]
        if input[corrupted[i]-1] in random:
            random.remove(input[corrupted[i]-1])
    return target_dict, random, target_re


def find_constrains(A, B, C, D):
    # Step 1: find which corrupt parties own the same invitee
    receiver_to_corrupted_parties = {}
    for party, receiver in C.items():
        if receiver in receiver_to_corrupted_parties:
            receiver_to_corrupted_parties[receiver].append(party)
        else:
            receiver_to_corrupted_parties[receiver] = [party]

    result = []
    # Step 2: Judge each set of data to see if there are any honest parties sending invitations to the receiver
    for receiver, corrupted_parties in receiver_to_corrupted_parties.items():
        non_corrupted_senders = any(A[i] == receiver and (i + 1) not in corrupted_parties for i in range(len(A)))
        result.append(corrupted_parties + [not non_corrupted_senders])

    # Step 3: after each item, add the number of corrupted inviters of the invitee
    for group in result:
        group.append(len(group) - 1)  # group[-1] is True/False，we don't need to count that
    # Step 4: after each item, add the number of inviters of the invitee
    for group in result:
        receiver = C[group[0]]
        group.append(len(D[receiver - 1]))
    # Step 5: Replace the numbers in each list with their position in the corrupted list
    B_positions = {party: index + 1 for index, party in enumerate(B)}
    final_result = []
    for group in result:
        new_group = [B_positions[party] for party in group[:-3]]
        new_group.extend(group[-3:])
        final_result.append(new_group)

    return final_result


def generate_combinations(n, constraints):
    def is_valid_combination(combination):
        for constraint in constraints:
            positions = constraint[:-3]
            required_value = constraint[-3]

            true_count = sum(combination[i - 1] for i in positions)

            if required_value:
                # There is and only one is True
                if true_count != 1:
                    return False
            else:
                # At most one is True (or all can be False)
                if true_count > 1:
                    return False
        return True

    def generate_all_combinations(n):
        if n == 0:
            return [[]]

        previous_combinations = generate_all_combinations(n - 1)
        result = []
        for combination in previous_combinations:
            result.append(combination + [True])
            result.append(combination + [False])

        return result

    all_combinations = generate_all_combinations(n)
    valid_combinations = [combo for combo in all_combinations if is_valid_combination(combo)]

    return valid_combinations


def check_elements(A, B, a):
    # A is the set of corrupted parties
    # B is the set of the inviters of a specific invitees
    # a is a corrupted parties and is one of the inviters in B
    # the usage of the function: check whether there is still other corrupted inviters in B
    # and calculate the honest parties in B
    other_elements_in_B = [element for element in A if element != a and element in B]
    elements_not_in_A = [element for element in B if element not in A]
    return len(other_elements_in_B) > 0, len(elements_not_in_A)


def calculate_probability(combinations, target_dict, target_re, corrupted, random_length, constrain):
    for i in range(len(combinations)):
        pr = 1
        for j in range(len(combinations[i])):
            if combinations[i][j]:
                cor = corrupted[j]
                target = target_dict[cor]
                # print(cor, target)
                possibility = 1 / len(target_re[target-1])
                pr *= possibility

        # After iterating through all terms of a possibility, check the condition
        for constraint in constrain:
            positions = constraint[:-3]
            required_value = constraint[-3]

            if not required_value:
                # Check whether the corresponding entries of positions are all False
                if all(combinations[i][pos-1] == False for pos in positions):
                    possibility = 1 - (constraint[-2] / constraint[-1])
                    pr *= possibility
        combinations[i].append(pr)
    return combinations


def generate_detailed_possibilities(input_list, corrupted, random, possibility):
    detailed_possibilities = []
    sum_p = 0

    for possible in possibility:
        base_distribution = [0] * len(corrupted)
        fixed_probability = possible[-1]

        # Step 1: assign the corrupted parties who have been accepted
        true_indices = [i for i, val in enumerate(possible[:-1]) if val]
        for index in true_indices:
            base_distribution[index] = input_list[corrupted[index] - 1]

        # Step 2: assign the corrupted parties who have not been accepted
        false_indices = [i for i, val in enumerate(possible[:-1]) if not val]
        available_receivers = random.copy()

        # Remove the assigned invitees
        for index in true_indices:
            if base_distribution[index] in available_receivers:
                available_receivers.remove(base_distribution[index])

        # Step 3: Generate all possible permutations
        for permutation in itertools.permutations(available_receivers, len(false_indices)):
            new_distribution = base_distribution.copy()
            for i, index in enumerate(false_indices):
                new_distribution[index] = permutation[i]

            # Step 4: calculate the probability
            probability = fixed_probability / math.perm(len(available_receivers), len(false_indices))
            sum_p += probability

            detailed_possibilities.append(new_distribution + [probability])
    # print("p:", sum_p)

    return detailed_possibilities


def print_detailed_possibilities(corrupted, detailed_possibilities):
    for possibility in detailed_possibilities:
        assignments = possibility[:-1]
        probability = possibility[-1]

        result_str = "Pr["
        for i in range(len(corrupted)):
            if i > 0:
                result_str += ", "
            result_str += f"O(P{corrupted[i]})=P{assignments[i]}"
        result_str += f"] = {probability}"

        print(result_str)


def calculate_leakage(input, corrupted):
    target_dict, random, target_re = calculate_target_random(input, corrupted)
    constrains = find_constrains(input, corrupted, target_dict, target_re)
    # print("con: ", constrains)
    combinations = generate_combinations(len(corrupted),constrains)
    # print(target_dict, random, target_re)
    # print(combinations)
    # print(len(combinations))
    distribution = calculate_probability(combinations, target_dict, target_re, corrupted, len(random),constrains)
    #print(distribution)
    detailed_possibilities = generate_detailed_possibilities(input, corrupted, random, distribution)
    print_detailed_possibilities(corrupted, detailed_possibilities)


# input = [1,2,3,1,2,3,1,2,3]
# corrupted = [1,2,3,6,8,9]
input = [2,1,2,2,1]
corrupted = [1,2]
calculate_leakage(input,corrupted)




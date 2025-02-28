import random


def redistribute_invitations(I):
    """
    Input:
        I: A 2D list of size n*d, where I[i][j] indicates the recipient of the j-th invitation sent by user Pi (user indices from 0 to n-1)
    Output:
        O: A 2D list of size n*d, where O[i] represents the list of sender indices whose invitations user Pi finally receives
    """
    n = len(I)  # Total number of users
    d = len(I[0])  # Number of invitations each user sends, and the final number each user should receive

    # 1. Build a list of invitations received by each user
    received = [[] for _ in range(n)]
    for sender in range(n):
        for target in I[sender]:
            received[target].append(sender)
    print(received)
    # 2. Preliminary assignment: if the number of invitations received is <= d, accept all;
    #    if more than d, randomly select d invitations to accept and mark the rest as surplus.
    accepted = [None] * n  # Final accepted invitation list for each user (records sender indices)
    surplus = []  # List of surplus invitations (records sender indices)

    for user in range(n):
        if len(received[user]) <= d:
            accepted[user] = list(received[user])
        else:
            # Randomly shuffle the received invitations before selecting d of them
            random.shuffle(received[user])
            accepted[user] = received[user][:d]
            surplus.extend(received[user][d:])

    # 3. Calculate the deficit (number of invitations needed) for each user and create a deficit list
    #    where each user appears as many times as invitations they are missing.
    deficit_list = []
    for user in range(n):
        deficit = d - len(accepted[user])
        deficit_list.extend([user] * deficit)

    # Check if the number of surplus invitations matches the total deficit
    if len(surplus) != len(deficit_list):
        raise Exception("The number of surplus invitations does not match the deficit count!")

    # 4. Randomly shuffle both the surplus and deficit lists, then pair them to assign surplus invitations
    #    to users with a deficit.
    random.shuffle(surplus)
    random.shuffle(deficit_list)

    for s, user in zip(surplus, deficit_list):
        accepted[user].append(s)

    return accepted


# Example usage
if __name__ == '__main__':
    # Example: Assume there are 3 users, each sending 2 invitations.
    # Input matrix I[i][j] indicates the recipient of the invitation sent by user Pi.
    I = [
        [1, 1],  # User 0 sends invitations to users 0 and 1
        [1, 2],  # User 1 sends invitations to users 1 and 2
        [2, 0]  # User 2 sends invitations to users 2 and 0
    ]
    O = redistribute_invitations(I)
    print("Final invitation senders for each user:")
    for i, invites in enumerate(O):
        print(f"User {i}: {invites}")
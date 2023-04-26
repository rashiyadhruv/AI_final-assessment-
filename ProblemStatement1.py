# Define constants

CONVERGENCE_THRESHOLD = 1e-3

DECIMAL_PLACES = 2

# Define the Bellman's Equation for value iteration

def get_expected_reward(curr_state, action):

    if action == 'stop':

        return reward[curr_state]['stop']

    else:

        expected_reward = 0

        for next_state in transition_probability[curr_state][action].keys():

            expected_reward += transition_probability[curr_state][action][next_state] * (reward[curr_state]['correct'] + gamma * V[next_state])

        return expected_reward

# Run the value iteration algorithm

for i in range(1000):

    delta = 0

    for s in states:

        old_v = V[s]

        max_action_value = -float('inf')

        for a in actions:

            action_value = get_expected_reward(s, a)

            if action_value > max_action_value:

                max_action_value = action_value

        V[s] = max_action_value

        delta = max(delta, abs(old_v - V[s]))

    if delta < CONVERGENCE_THRESHOLD:

        break

# Print the optimal values rounded to DECIMAL_PLACES decimal places

print('Optimal Values:')

print('State\tOptimal Value')

for s in states[:-1]:

    print(f'{s}\t{round(V[s], DECIMAL_PLACES)}')

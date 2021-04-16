import numpy as np


class ValueIteration:
    """
    Value iteration for a Markov Decision Process (MDP)
    """

    def __init__(self, states, actions, transition_key):
        self.values = None
        self.policy = None
        self.states = states
        self.actions = actions
        self.transition_key = transition_key

    def fit(self, iterations=100, gamma=0.9, verbose=False):
        """
        Value Iteration Algorithm - Learn Bellman optimality values
        """

        values = [0] * len(self.states)
        max_temp = [[0]] * len(self.states)

        for i in range(iterations):
            # loop through starting states
            for s1, state1 in enumerate(self.states):
                if verbose:
                    print(f'##### {i} State {state1} #####')

                # List to hold all the bellman values for each transition
                list_temp = []

                # loop through actions
                for act in self.actions:
                    # store the bellman value for this action. if an action has multiple branches the values
                    # need to be accumulated
                    value_temp = 0

                    # loop through final states
                    for s2, state2 in enumerate(self.states):
                        # Get transition probability and reward
                        if (state1, act, state2) in self.transition_key:
                            prob, reward = self.transition_key[(state1, act, state2)]
                            value_temp += prob * (reward + (gamma * values[s2]))

                            if verbose:
                                print(f'transition: {state1} {act} {state2}')
                                print(f'prob: {prob}')
                                print(f'reward: {reward}')
                                print(f'temp value: {value_temp}')

                    if verbose:
                        print()

                    # append the bellman value for this action to the list
                    list_temp.append(value_temp)

                # append to list containing bellman values for all actions across both states
                max_temp[s1] = list_temp

            # calculate new value for each state by taking max
            values = np.max(max_temp, axis=1)

            # calculate policy for each state by taking argmax
            policy = np.argmax(max_temp, axis=1)

            if verbose:
                print(f'list of bellman values: {max_temp}')
                print(f'list of max values: {values} \n')

        self.values = values
        self.policy = policy

    def play(self, state):
        """
        Given a state, return the chosen action for this state
        """

        # return the policy action for a given state
        state_index = self.states.index(state)
        policy_index = self.policy[state_index]

        return self.actions[policy_index]


# Example use case: Robot vacuum cleaner
if __name__ == "__main__":
    # States are high and low battery
    states = ['high', 'low']
    # Actions are to search, wait or recharge
    actions = ['search', 'wait', 'recharge']
    # Dict with key (state1, action, state2) and values (probability, reward)
    transition_key = {
        ('high', 'search', 'low'): (0.4, 5),
        ('high', 'search', 'high'): (0.6, 5),
        ('high', 'wait', 'high'): (1, 1),
        ('low', 'wait', 'low'): (1, 1),
        ('low', 'search', 'high'): (0.5, -3),
        ('low', 'search', 'low'): (0.5, 5),
        ('low', 'recharge', 'high'): (1, 0)
    }

    # Create value iteration instance and learn optimal policy
    vi = ValueIteration(states, actions, transition_key)
    vi.fit(iterations=100, gamma=0.9, verbose=False)
    print(f'values: {vi.values}, policy {vi.policy}')

    # Get action from policy for a given state
    print(f'In high state, optimal policy is to: {vi.play("high")}')
    print(f'In low state, optimal policy is to: {vi.play("low")}')
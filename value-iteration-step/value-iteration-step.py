import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    V = np.array(values)
    T = np.array(transitions)
    R = np.array(rewards)

    expected_future = np.sum(T*V, axis=2)

    discounted_future = R + gamma * expected_future

    new_values = np.max(discounted_future, axis=1)

    return new_values.tolist()

"""
Shared training data and RBF centers for CartPole experiments.
All files should import from here to ensure consistency across experiments.
"""
import random
import numpy as np
import jax.numpy as jnp
from cartpole import CartPole

# Fixed parameters
N_TRAIN = 2000
M_CENTERS = 1000
RANDOM_SEED = 42


def generate_training_data(N=N_TRAIN, with_action=False):
    """
    Generate training data for CartPole dynamics.

    Args:
        N: Number of training samples
        with_action: If True, includes action as 5th input dimension

    Returns:
        X: Input states, shape (N, 4) or (N, 5) if with_action
        Y: Output state deltas, shape (N, 4)
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_list = []
    Y_list = []

    for _ in range(N):
        example_system = CartPole(visual=False)

        cart_position = random.uniform(-2.5, 2.5)
        cart_velocity = random.uniform(-12, 12)
        pole_angle = random.uniform(-np.pi, np.pi)
        pole_velocity = random.uniform(-30, 30)

        if with_action:
            action = random.uniform(-20, 20)
            state = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])
            X_list.append([cart_position, cart_velocity, pole_angle, pole_velocity, action])
            example_system.setState(state.tolist())
            example_system.performAction(action)
        else:
            state = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])
            X_list.append(state)
            example_system.setState(state.tolist())
            example_system.performAction()

        current_state = np.array(example_system.getState())
        Y_list.append(current_state - state)

    X = jnp.array(X_list)
    Y = jnp.array(Y_list)

    return X, Y


def generate_rbf_centers(M=M_CENTERS, with_action=False):
    """
    Generate RBF centers for kernel-based models.

    Args:
        M: Number of RBF centers
        with_action: If True, includes action as 5th dimension

    Returns:
        T: RBF centers, shape (M, 4) or (M, 5) if with_action
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    T_list = []
    for _ in range(M):
        cart_position = random.uniform(-2.5, 2.5)
        cart_velocity = random.uniform(-12, 12)
        pole_angle = random.uniform(-np.pi, np.pi)
        pole_velocity = random.uniform(-30, 30)

        if with_action:
            action = random.uniform(-20, 20)
            T_list.append([cart_position, cart_velocity, pole_angle, pole_velocity, action])
        else:
            T_list.append([cart_position, cart_velocity, pole_angle, pole_velocity])

    T = jnp.array(T_list)
    return T


def get_standard_data():
    """
    Get the standard training data and centers (no action).

    Returns:
        X: Training inputs, shape (2000, 4)
        T: RBF centers, shape (1000, 4)
        Y: Training outputs, shape (2000, 4)
    """
    X, Y = generate_training_data(N=N_TRAIN, with_action=False)
    T = generate_rbf_centers(M=M_CENTERS, with_action=False)
    return X, T, Y


def get_action_data():
    """
    Get training data and centers with action as input.

    Returns:
        X: Training inputs, shape (2000, 5) - includes action
        T: RBF centers, shape (1000, 5) - includes action
        Y: Training outputs, shape (2000, 4)
    """
    X, Y = generate_training_data(N=N_TRAIN, with_action=True)
    T = generate_rbf_centers(M=M_CENTERS, with_action=True)
    return X, T, Y


if __name__ == "__main__":
    # Test the data generation
    print("Generating standard data (no action)...")
    X, T, Y = get_standard_data()
    print(f"X shape: {X.shape}, T shape: {T.shape}, Y shape: {Y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Y range: [{Y.min():.2f}, {Y.max():.2f}]")

    print("\nGenerating action data...")
    X_action, T_action, Y_action = get_action_data()
    print(f"X shape: {X_action.shape}, T shape: {T_action.shape}, Y shape: {Y_action.shape}")
    print(f"Action range: [{X_action[:, 4].min():.2f}, {X_action[:, 4].max():.2f}]")

    print("\nData generation complete!")
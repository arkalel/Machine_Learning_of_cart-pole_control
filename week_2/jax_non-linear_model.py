import random
import scipy.optimize as scopt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cartpole import CartPole, remap_angle


@jax.jit
def rbf_kernel(A, B, omega):
    """Gaussian RBF kernel; A is (N, 4), B is (M, 4), returns (N, M)."""
    diff = A[:, None, :] - B[None, :, :]
    exponent = (
        diff[..., 0] ** 2 / omega[0] ** 2
        + diff[..., 1] ** 2 / omega[1] ** 2
        + jnp.sin(diff[..., 2] / 2) ** 2 / omega[2] ** 2
        + diff[..., 3] ** 2 / omega[3] ** 2
    )
    return jnp.exp(-exponent)

@jax.jit
def fit_alphas(K, KMM, Y, lambda_):
    """Solve (K.T @ K + lambda_ * KMM) @ alpha = K.T @ Y for all output dims."""
    A = K.T @ K + lambda_ * KMM
    B = K.T @ Y
    return jnp.linalg.solve(A, B)

@jax.jit
def predict_delta(state, T, omega, alpha):
    """State transition delta from RBF model; state is (4,), alpha is (M, 4)."""
    k_pred = rbf_kernel(state[None, :], T, omega)[0]
    return k_pred @ alpha

omega1 = 1000
omega2 = 5.4
omega3 = 0.8
omega4 = 4.6
lambda_ = 0.00001
omega = jnp.array([omega1, omega2, omega3, omega4])

n = 500
M = 640
N = 1280
z = 1
X_list = []
Y_list = []
#training data
for _ in range(N):
    example_system = CartPole(visual=False)
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-10, 10)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-15, 15)
    state = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])
    X_list.append(state)
    example_system.setState(state.tolist())
    example_system.performAction()
    current_state = np.array(example_system.getState())
    Y_list.append(current_state - state)
X = jnp.array(X_list)
Y = jnp.array(Y_list)
# getting the centers for the RBFs
T_list = []
for _ in range(M):
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-10, 10)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-15, 15)
    T_list.append(
        [cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity]
    )
T = jnp.array(T_list)

K = rbf_kernel(X, T, omega)
KMM = rbf_kernel(T, T, omega)
alpha = fit_alphas(K, KMM, Y, lambda_)

X_rollout = np.empty((n, 4), float)
pred_X = np.empty((n, 4), float)
example_system = CartPole(visual=False)
cart_position = random.uniform(-2.5, 2.5)
cart_velocity = random.uniform(-10, 10)
pole_angle = np.random.uniform(-np.pi, np.pi)
pole_velocity = random.uniform(-15, 15)
state = [cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity]
example_system.setState(state)
pred_state = jnp.array(state)
#print("state is ", state)
for i in range(n):
    state = example_system.getState()
    X_rollout[i] = state
    pred_X[i] = np.asarray(pred_state)
    example_system.performAction()
    delta = predict_delta(pred_state, T, omega, alpha)
    pred_state = pred_state + delta
    #print("you added is ", delta)
    #print("pred state is ", pred_state)
x_stream0 = X_rollout[:, 0]
x_stream1 = X_rollout[:, 1]
x_stream2 = X_rollout[:, 2]
x_stream3 = X_rollout[:, 3]
pred_x_stream0 = pred_X[:, 0]
pred_x_stream1 = pred_X[:, 1]
pred_x_stream2 = pred_X[:, 2]
pred_x_stream3 = pred_X[:, 3]
residual = X_rollout - pred_X
mse = jnp.mean(residual ** 2)

plt.plot(x_stream0, label="True cart position")
plt.plot(pred_x_stream0, label="Predicted cart position")
plt.plot(x_stream1, label="True cart velocity")
plt.plot(pred_x_stream1, label="Predicted cart velocity")
plt.plot(x_stream2, label="True pole angle")
plt.plot(pred_x_stream2, label="Predicted pole angle")
plt.plot(x_stream3, label="True pole velocity")
plt.plot(pred_x_stream3, label="Predicted pole velocity")
plt.xlabel("Time Step")
plt.ylabel("Cart Position")
plt.legend()
plt.show()

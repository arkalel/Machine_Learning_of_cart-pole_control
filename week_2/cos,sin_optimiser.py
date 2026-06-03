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
    sin_diff = jnp.sin(A[:, None, :]/2) - jnp.sin(B[None, :, :]/2)
    cos_diff = jnp.cos(A[:, None, :]/2) - jnp.cos(B[None, :, :]/2)
    exponent = (
        diff[..., 0] ** 2 / omega[0] ** 2
        + diff[..., 1] ** 2 / omega[1] ** 2
        + jnp.sin(diff[..., 2]) ** 2 / omega[2] ** 2
        + diff[..., 3] ** 2 / omega[3] ** 2
        - jnp.cos(diff[..., 2]) ** 2 / omega[4] ** 2
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
    
def get_mse(parameters):
    #X, T, Y = get_variables()
    omega1 = parameters[0]
    omega2 = parameters[1]
    omega3 = parameters[2]
    omega4 = parameters[3]
    omega5 = parameters[5]
    lambda_ = parameters[4]
    omega = jnp.array([omega1, omega2, omega3, omega4, omega5])

    K = rbf_kernel(X, T, omega)
    KMM = rbf_kernel(T, T, omega)
    alpha = fit_alphas(K, KMM, Y, lambda_)
    rollout_mse_values = []

    for j in range(num_starts):
        cp = start_cps[j]
        cv = start_cvs[j]
        pa = start_pas[j]
        pv = start_pvs[j]
        state = [cp, cv, remap_angle(pa), pv]
        X_rollout = np.empty((n, 4), float)
        pred_X = np.empty((n, 4), float)
        example_system = CartPole(visual=False)
        state = [cp, cv , pa, pv]
        example_system.setState(state)
        pred_state = jnp.array(state)
        #print("state is ", state)
        rollout_mse = 0
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
            rollout_mse = jnp.mean(residual ** 2)
        
        rollout_mse_values.append(rollout_mse)
    return float(np.mean(rollout_mse_values))



def get_averaged_mse(parameters):
    training_mse_values = []
    for i in range(1):
        training_mse_values.append(get_mse(parameters))
    print("parameters are ", parameters)
    print("mean training mse is ", float(np.mean(training_mse_values)))
    return float(np.mean(training_mse_values))

def get_train_mse(parameters):
    omega = jnp.array([parameters[0], parameters[1], parameters[2],
                       parameters[3], parameters[5]])
    lambda_ = parameters[4]
    K = rbf_kernel(X, T, omega)
    KMM = rbf_kernel(T, T, omega)
    alpha = fit_alphas(K, KMM, Y, lambda_)
    Y_pred = K @ alpha
    return float(jnp.mean((Y - Y_pred) ** 2))

n = 100
M = 1280
N = 2560
z = 1
num_starts = 10
start_cps = [0] * num_starts 
start_cvs = [0] * num_starts
start_pas = [0] * num_starts
start_pvs = [0] * num_starts

for i in range(num_starts):
    start_cps[i] = random.uniform(-2.5, 2.5)
    start_cvs[i] = random.uniform(-10, 10)
    start_pas[i] = random.uniform(-np.pi, np.pi)
    start_pvs[i] = random.uniform(-15, 15)

#def get_variables():
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
#return X, T, Y

start_parameters = np.array([900, 11, 2.1 , 11 , 0.000001, 2.1])
bounds = ((10, 10000), (1, 50), (0.1, 10), (1, 50), (1e-6, 0.1), (1,10))
eps =(0.005 * start_parameters)
result = scopt.minimize(get_averaged_mse, start_parameters, method='L-BFGS-B',
                        bounds=bounds, options={"maxiter": 200, "eps": eps, "ftol": 1e-5})
print("result is ", result)
print("best params [w1,w2,w3,w4,lambda,w5]:", result.x)
print("best mse:", result.fun)
print("rollout mse:", get_mse(result.x))
#best params [w1,w2,w3,w4,lambda,w5]: [8.99904787e+02 1.16986915e+01 2.11185866e+00 1.11728154e+01 8.32273336e-06 2.11185866e+00]
#best mse: 557.9356689453125
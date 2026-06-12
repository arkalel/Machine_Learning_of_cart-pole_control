import random
import scipy.optimize as scopt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cartpole import CartPole, remap_angle
from training_data import get_action_data


@jax.jit
def policy(state, P):
    return jnp.dot(P, state)


#@jax.jit
def loss(X):
    omegal = 40.0
    exponent = 0
    exponent = (0.005 * (X[..., 0] ** 2 / (5 ** 2)
    + X[..., 1] ** 2/ (5 ** 2)
    + (X[..., 2]) ** 2 /(1 ** 2)
    + X[..., 3] ** 2 /(1 ** 2)
    ))
    #print('contributions are ', [10 * (X[..., 0] ** 2) / (5 ** 2), 10 * (X[..., 1] ** 2) / (5 ** 2), 10 * (X[..., 2] ** 2) / (1 ** 2), 10 * (X[..., 3] ** 2) / (1 ** 2)])
    #print((1 - jnp.exp(-exponent)).mean())
    return (1.0 - jnp.exp(-exponent)).mean()

@jax.jit
def rbf_kernel(A, B, omega):
    """Gaussian RBF kernel; A is (N, 4), B is (M, 4), returns (N, M)."""
    diff = A[:, None, :] - B[None, :, :]
    sin_diff = jnp.sin(A[:, None, :]) - jnp.sin(B[None, :, :])
    cos_diff = jnp.cos(A[:, None, :]) - jnp.cos(B[None, :, :])
    exponent = (
        diff[..., 0] ** 2 / omega[0] ** 2
        + diff[..., 1] ** 2 / omega[1] ** 2
        + sin_diff[..., 2] ** 2 / omega[2] ** 2
        + diff[..., 3] ** 2 / omega[3] ** 2
        + cos_diff[..., 2] ** 2 / omega[4] ** 2
        + (A[:, None, 4] - B[None, :, 4]) ** 2 / omega[5] ** 2
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

def get_variables():
    X_list = []
    Y_list = []
    #training data
    for _ in range(N):
        example_system = CartPole(visual=False)
        cart_position = random.uniform(-2.5, 2.5)
        cart_velocity = random.uniform(-10, 10)
        pole_angle = random.uniform(-np.pi, np.pi)
        pole_velocity = random.uniform(-25, 25)
        action = random.uniform(-20, 20)
        state = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])
        X_list.append([cart_position, cart_velocity, pole_angle, pole_velocity, action])
        example_system.setState(state.tolist())
        example_system.performAction(action)
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
        pole_velocity = random.uniform(-25, 25)
        action = random.uniform(-20, 20)
        T_list.append(
            [cart_position, cart_velocity, pole_angle, pole_velocity, action]
        )
    T = jnp.array(T_list)
    return X, T, Y

def rollout_graph(X, pred_X):

    x_stream0 = []
    x_stream1 = []
    x_stream2 = []
    x_stream3 = []
    pred_x_stream0 = []
    pred_x_stream1 = []
    pred_x_stream2 = []
    pred_x_stream3 = []
    action_stream = []
    time = []

    for j in range(n):

        time.append(j*0.1)
        
        x_stream0.append(X[j][0])
        x_stream1.append(X[j][1])
        x_stream2.append(X[j][2])
        x_stream3.append(X[j][3])
        pred_x_stream0.append(pred_X[j][0])
        pred_x_stream1.append(pred_X[j][1])
        pred_x_stream2.append(pred_X[j][2])
        pred_x_stream3.append(pred_X[j][3])
        action_stream.append(pred_X[j][4])

    plt.plot(time, x_stream0, label='cart position')
    plt.plot(time, x_stream1, label='cart velocity')
    plt.plot(time, x_stream2, label='pole angle')
    plt.plot(time, x_stream3, label='pole velocity')
    #plt.plot(time, pred_x_stream0, label='Predicted position', linestyle='dashed', color = 'blue')
    #plt.plot(time, pred_x_stream1, label='Predicted velocity', linestyle='dashed', color = 'orange')
    #plt.plot(time, pred_x_stream2, label='Predicted angle', linestyle='dashed', color = 'green')
    #plt.plot(time, pred_x_stream3, label='Predicted pole velocity', linestyle='dashed', color = 'red')
    plt.plot(time, action_stream, label= 'force')

    plt.xlabel('time')
    plt.ylabel('state')
    plt.legend()
    plt.show()

def simulate_rollout_and_loss(P):
    total_loss = 0


    for j in range(num_starts):
        cp = start_cps[j]
        cv = start_cvs[j]
        pa = start_pas[j]
        pv = start_pvs[j]
        state = [cp, cv, pa, pv]
        X_rollout = np.empty((n, 5), float)
        pred_X = np.empty((n, 5), float)
        example_system = CartPole(visual=False)
        state = [cp, cv , pa, pv]
        example_system.setState(state)
        pred_state = jnp.array(state)
        rollout_loss = 0
        
        for i in range(n):
            loss_value = 0
            state = example_system.getState()
            action = policy(state, P)
            X_rollout[i] = [state[0], state[1], state[2], state[3], action]
            pred_X[i] = [pred_state[0], pred_state[1], pred_state[2], pred_state[3],action]
            example_system.performAction(action)
            total_state = np.array([pred_state[0], pred_state[1], pred_state[2], pred_state[3], action])
            delta = predict_delta(total_state, T, omega, alpha)
            pred_state = pred_state + delta

        
        rollout_loss = loss(pred_X)
        #print('rollout loss is ', rollout_loss)
        total_loss += rollout_loss

    print('P is ', P)
    rollout_graph(X_rollout, pred_X)
    print('average loss is ', total_loss / (num_starts))
    return total_loss / (num_starts)
        

n = 800
M = 1500
N = 3000
#lambda_ = 0.000025
#omega1 = 900
#omega2 = 5
#omega3 = 0.8
#omega4 = 6
#omega5 = 0.8
#omega6 = 7.1
lambda_ = 0.000025
omega1 = 900      #[15.80696907 18.26257509 11.49651202  1.32678698  5.42532521]
omega2 = 8.4
omega3 = 0.6
omega4 = 7
omega5 = 0.6
omega6 = 10

X, T, Y = get_variables()
#X, T, Y = get_action_data()
num_starts = 1
start_cps = [0] * num_starts 
start_cvs = [0] * num_starts
start_pas = [0] * num_starts
start_pvs = [0] * num_starts

for i in range(num_starts):
    start_cps[i] = 0
    start_cvs[i] = 0
    start_pas[i] = 0.1
    start_pvs[i] = 0

omega = np.array([omega1, omega2, omega3, omega4, omega5, omega6])
K = rbf_kernel(X, T, omega)
KMM = rbf_kernel(T, T, omega)
alpha = fit_alphas(K, KMM, Y, lambda_)

P = np.array([-1.28, 6.33, 12.51, 6.30])  #[-1.28219168  6.33260175 12.51113255  6.3039672 ] is best for keeping upright, with average loss of  0.000102930666
#[-44, -7, 8.7, 0.3] best for start with pa = pi
print(simulate_rollout_and_loss(P))


start_parameters = np.array([-1.28219168,  6.33260175, 12.51113255,  6.3039672 ])
bounds = ((-1000, 1000), (-1000, 1000), (-1000, 1000), (-1000, 1000))
eps =(0.01 * start_parameters)
result = scopt.minimize(simulate_rollout_and_loss, start_parameters, method='L-BFGS-B',
                        bounds=bounds, options={"maxiter": 200, "eps": eps, "ftol": 1e-5})
print("result is ", result)
print("best params [a,b,c,d]:", result.x)
print("best loss:", result.fun)
print("best rollout loss:", simulate_rollout_and_loss(result.x))
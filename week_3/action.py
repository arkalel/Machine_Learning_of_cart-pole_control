import random
from time import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cartpole import CartPole, remap_angle

n = 1000
M = 1000
N = 2000
lambda_ = 0.0006  
omega1 = 900
omega2 = 7.4
omega3 = 0.56
omega4 = 3.45
omega5 = 1.00
omega6 = 10.0
omega = np.array([omega1, omega2, omega3, omega4, omega5, omega6])
K = np.zeros((N, M))
KMM = np.zeros((M, M))
exponent = 0
# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
X = np.empty((0, 5), float)
x_cart_position = []
x_cart_velocity = []
x_pole_angle = []
x_pole_velocity = []
Y = np.empty((0, 4), float)
d_cart_position = []
d_cart_velocity = []
d_pole_angle = []
d_pole_velocity = []
y1 = np.empty(N, float)
y2 = np.empty(N, float)
y3 = np.empty(N, float)
y4 = np.empty(N, float)

#getting training data
for i in range(N):
    example_system = CartPole(visual=False)
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-12, 12)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-18, 18)
    action = random.uniform(-10, 10)

    state = [[cart_position, cart_velocity, pole_angle, pole_velocity]]
    X = np.append(X, [[cart_position, cart_velocity, pole_angle, pole_velocity, action]], axis=0)

    example_system.setState(state[0])
    example_system.performAction(action)

    current_state = [example_system.getState()]
    current_state = [[current_state[0][0], current_state[0][1], current_state[0][2], current_state[0][3], action]]
    Y = np.append(Y, [[current_state[0][0]- state[0][0], current_state[0][1]-state[0][1], current_state[0][2]-state[0][2], current_state[0][3]-state[0][3]]], axis=0)
    y1[i] = current_state[0][0]- state[0][0]
    y2[i] = current_state[0][1]- state[0][1]
    y3[i] = current_state[0][2]- state[0][2]
    y4[i] = current_state[0][3]- state[0][3]

#getting the centers for the RBFs
T = np.empty((M, 5), float)
for i in range(M):
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-15, 15)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-20, 20)
    action = random.uniform(-10, 10)

    T[i] = [cart_position, cart_velocity, pole_angle, pole_velocity, action]


for i in range(N):
    for j in range(M):
        exponent = 0
        exponent = exponent + (X[i][0] - T[j][0]) ** 2 / omega[0] ** 2
        exponent = exponent + (X[i][1] - T[j][1]) ** 2 / omega[1] ** 2
        exponent = exponent + (np.sin((X[i][2] - T[j][2])/2)) ** 2  / omega[2] ** 2
        exponent = exponent + (X[i][3] - T[j][3]) ** 2 / omega[3] ** 2
        exponent = exponent - (np.cos((X[i][2] - T[j][2])/2)) ** 2 / omega[4] ** 2
        exponent = exponent + (X[i][4] - T[j][4]) ** 2 / omega[5] ** 2
        K[i][j] = np.exp(-exponent)

for i in range(M):
    for j in range(M):

        exponent = 0
        exponent = exponent + (T[i][0] - T[j][0]) ** 2 / omega[0] ** 2
        exponent = exponent + (T[i][1] - T[j][1]) ** 2 / omega[1] ** 2
        exponent = exponent + (np.sin((T[i][2] - T[j][2])/2)) ** 2  / omega[2] ** 2
        exponent = exponent + (T[i][3] - T[j][3]) ** 2 / omega[3] ** 2
        exponent = exponent - np.cos((T[i][2] - T[j][2])/2) ** 2 / omega[4] ** 2
        exponent = exponent + (T[i][4] - T[j][4]) ** 2 / omega[5] ** 2
        KMM[i][j] = np.exp(-exponent)


#alpha1 = np.linalg.inv(K.T @ K + lambda_ * KMM) @ K.T @ y1
#alpha2 = np.linalg.inv(K.T @ K + lambda_ * KMM) @ K.T @ y2
#alpha3 = np.linalg.inv(K.T @ K + lambda_ * KMM) @ K.T @ y3
#alpha4 = np.linalg.inv(K.T @ K + lambda_ * KMM) @ K.T @ y4
alpha1_t = np.linalg.lstsq((K.T @ K + lambda_ * KMM).T, (K.T @ y1).T, rcond=None)[0]
alpha2_t = np.linalg.lstsq((K.T @ K + lambda_ * KMM).T, (K.T @ y2).T, rcond=None)[0]
alpha3_t = np.linalg.lstsq((K.T @ K + lambda_ * KMM).T, (K.T @ y3).T, rcond=None)[0]
alpha4_t = np.linalg.lstsq((K.T @ K + lambda_ * KMM).T, (K.T @ y4).T, rcond=None)[0]
alpha1 = alpha1_t.T
alpha2 = alpha2_t.T
alpha3 = alpha3_t.T
alpha4 = alpha4_t.T

print('alpha1 is ', alpha1)
print('alpha2 is ', alpha2)
print('alpha3 is ', alpha3)
print('alpha4 is ', alpha4)

Y_pred = np.empty((N, 4), float)
Y_pred = K @ np.array([alpha1, alpha2, alpha3, alpha4]).T
print('Y_pred is ', Y_pred)
plt.plot(Y[:,2], label='True change in pole angle')
plt.plot(Y_pred[:,2], label='Predicted change in pole angle', linestyle='dashed')
plt.xlabel('time step')
plt.ylabel('change in pole angle')
plt.legend()
plt.show()

def policy(state):
        P = np.array([-44, -7, 8.7, 0.3])
        return P.dot(state)

X = np.empty((n, 4), float)
pred_X = np.empty((n, 4), float)
x_cart_position = []
x_cart_velocity = []
x_pole_angle = []
x_pole_velocity = []
x_action = []
Y = np.empty((n, 4), float)
pred_Y = np.empty((n, 4), float)
d_cart_position = []
d_cart_velocity = []
d_pole_angle = []
d_pole_velocity = []

example_system = CartPole(visual=False)
cart_position = random.uniform(-2.5, 2.5)
cart_velocity = random.uniform(-10, 10)
pole_angle = random.uniform(-np.pi, np.pi)
pole_velocity = random.uniform(-15, 15)
action = random.uniform(-10, 10)

state = [cart_position, cart_velocity, pole_angle, pole_velocity]
example_system.setState(state)
pred_state = np.array(state)
print('state is ', state)
K_pred = np.empty(M, float)

for i in range(n):
    action = random.uniform(-10, 10)
    state = example_system.getState()
    X[i] = state
    pred_X[i] = pred_state
    
    example_system.performAction(action)

    current_state = example_system.getState()
    
    for j in range(M):
        exponent = 0
        exponent = exponent + ((pred_state[0] - T[j][0]) ** 2) / (omega[0] ** 2)
        exponent = exponent + ((pred_state[1] - T[j][1]) ** 2) / (omega[1] ** 2)
        exponent = exponent + (np.sin((pred_state[2] - T[j][2])/2)) ** 2  / omega[2] ** 2
        exponent = exponent + (pred_state[3] - T[j][3]) ** 2 / omega[3] ** 2
        exponent = exponent - np.cos((pred_state[2] - T[j][2])/2) ** 2 / omega[4] ** 2
        exponent = exponent + (action - T[j][4]) ** 2 / omega[5] ** 2
        K_pred[j] = np.exp(-exponent)
    pred_state = pred_state + K_pred @ np.array([alpha1, alpha2, alpha3, alpha4]).T
    print('you added is ', K_pred @ np.array([alpha1, alpha2, alpha3, alpha4]).T)
    print('pred state is ', pred_state)

y_stream0 = []
y_pred_stream0 = []
y_stream1 = []
y_pred_stream1 = []
y_stream2 = []
y_pred_stream2 = []
y_stream3 = []
y_pred_stream3 = []
x_stream0 = []
x_stream1 = []
x_stream2 = []
x_stream3 = []
pred_x_stream0 = []
pred_x_stream1 = []
pred_x_stream2 = []
pred_x_stream3 = []
time = []

for j in range(n):

    time.append(j*0.1)
    y_stream0.append(Y[j][0])
    y_pred_stream0.append(pred_Y[j][0])
    y_stream1.append(Y[j][1])
    y_pred_stream1.append(pred_Y[j][1])
    y_stream2.append(Y[j][2])
    y_pred_stream2.append(pred_Y[j][2])
    y_stream3.append(Y[j][3])
    y_pred_stream3.append(pred_Y[j][3])
    x_stream0.append(X[j][0])
    x_stream1.append(X[j][1])
    x_stream2.append(X[j][2])
    x_stream3.append(X[j][3])
    pred_x_stream0.append(pred_X[j][0])
    pred_x_stream1.append(pred_X[j][1])
    pred_x_stream2.append(pred_X[j][2])
    pred_x_stream3.append(pred_X[j][3])

plt.plot(time, x_stream0, label='cart position')
plt.plot(time, x_stream1, label='cart velocity')
plt.plot(time, x_stream2, label='pole angle')
plt.plot(time, x_stream3, label='pole velocity')
plt.plot(time, pred_x_stream0, label='Predicted position', linestyle='dashed')
plt.plot(time, pred_x_stream1, label='Predicted velocity', linestyle='dashed')
plt.plot(time, pred_x_stream2, label='Predicted angle', linestyle='dashed')
plt.plot(time, pred_x_stream3, label='Predicted pole velocity', linestyle='dashed')

plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()
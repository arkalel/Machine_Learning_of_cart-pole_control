import random
from time import time
import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole, remap_angle

n = 50
# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
X = np.empty((0, 4), float)
x_cart_position = []
x_cart_velocity = []
x_pole_angle = []
x_pole_velocity = []
Y = np.empty((0, 4), float)
d_cart_position = []
d_cart_velocity = []
d_pole_angle = []
d_pole_velocity = []

for i in range(500):
    example_system = CartPole(visual=False)
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-10, 10)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-15, 15)

    state = [[cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity]]
    X = np.append(X, state, axis=0)
    x_cart_position.append(state[0][0])
    x_cart_velocity.append(state[0][1])
    x_pole_angle.append(remap_angle(state[0][2]))
    x_pole_velocity.append(state[0][3])

    example_system.setState(state[0])
    example_system.performAction()

    current_state = [example_system.getState()]
    current_state = [[current_state[0][0], current_state[0][1], current_state[0][2], current_state[0][3]]]
    d_cart_position.append(current_state[0][0] - state[0][0])
    d_cart_velocity.append(current_state[0][1] - state[0][1])
    d_pole_angle.append(current_state[0][2] - state[0][2])
    d_pole_velocity.append(current_state[0][3] - state[0][3])
    Y = np.append(Y, [[current_state[0][0]- state[0][0], current_state[0][1]-state[0][1], current_state[0][2]-state[0][2], current_state[0][3]-state[0][3]]], axis=0)
#print(X)
#print(Y) 
C = Y.T @ X @ np.linalg.inv(X.T @ X)
print(C)
Y_pred = C @ X.T
#print(Y_pred.T[3:7])
#print(Y[3:7])

X = np.empty((n, 4), float)
pred_X = np.empty((n, 4), float)
x_cart_position = []
x_cart_velocity = []
x_pole_angle = []
x_pole_velocity = []
Y = np.empty((n, 4), float)
pred_Y = np.empty((n, 4), float)
d_cart_position = []
d_cart_velocity = []
d_pole_angle = []
d_pole_velocity = []

example_system = CartPole(visual=False)
cart_position = 0
cart_velocity = 0
pole_angle = 1
pole_velocity = 0

state = np.array([cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity])
example_system.setState(state)
pred_state = state
print(state[1])

for i in range(n):
    state = [example_system.getState()]
    X[i] = state[0]
    pred_X[i] = pred_state[0]
    #print('x is ',X)
    #print(Y)

    
    example_system.performAction()

    current_state = np.array([example_system.getState()])
    Y[i] = current_state - state
    pred_state_remapped = pred_state
    pred_state_remapped[2] = remap_angle(pred_state[2])
    print('pred state is ', pred_state_remapped)
    print('C@ pred state is ', C @ pred_state_remapped)
    print('pred state after adding is ', pred_state + (C @ pred_state_remapped))
    pred_state = pred_state + (C @ pred_state_remapped)
    pred_Y[i] = C @ pred_state


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
    #k = random.randint(0, 3)
    #n = random.randint(0, 499)
    time.append(j*10)
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
#print(X)
#plt.plot(x_stream0, label='cart position')
#plt.plot(x_stream1, label='cart velocity')
#plt.plot(x_stream2, label='pole angle')
plt.plot(x_stream3, label='pole velocity')
plt.scatter(pred_x_stream0, time, label='Predicted position', linestyle='dashed')
plt.scatter(pred_x_stream1, time, label='Predicted velocity', linestyle='dashed')
plt.scatter(pred_x_stream2, time, label='Predicted angle', linestyle='dashed')
plt.scatter(pred_x_stream3, time, label='Predicted pole velocity', linestyle='dashed')

plt.xlabel('time step')
plt.ylabel('state')
plt.legend()
plt.show()


import random
import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole, remap_angle

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
    #current_state = [[current_state[0][0], current_state[0][1], current_state[0][2], current_state[0][3]]]
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

for j in range(200):
    #k = random.randint(0, 3)
    n = random.randint(0, 499)
    y_stream0.append(Y[n][0])
    y_pred_stream0.append(Y_pred.T[n][0])
    y_stream1.append(Y[n][1])
    y_pred_stream1.append(Y_pred.T[n][1])
    y_stream2.append(Y[n][2])
    y_pred_stream2.append(Y_pred.T[n][2])
    y_stream3.append(Y[n][3])
    y_pred_stream3.append(Y_pred.T[n][3])
    x_stream0.append(X[n][0])
    x_stream1.append(X[n][1])
    x_stream2.append(X[n][2])
    x_stream3.append(X[n][3])

plt.scatter(x_stream1, y_stream0, label='change in Cart Position')
plt.scatter(x_stream1, y_stream1, label='change in Cart Velocity')
plt.scatter(x_stream1, y_stream2, label='change in Pole Angle')
plt.scatter(x_stream1, y_stream3, label='change in Pole Velocity')
plt.scatter(x_stream1, y_pred_stream0, label='Predicted change in Cart Position', linestyle='dashed')
plt.scatter(x_stream1, y_pred_stream1, label='Predicted change in Cart Velocity', linestyle='dashed')
plt.scatter(x_stream1, y_pred_stream2, label='Predicted change in Pole Angle', linestyle='dashed')
plt.scatter(x_stream1, y_pred_stream3, label='Predicted change in Pole Velocity', linestyle='dashed')
plt.xlabel('cart velocity')
plt.ylabel('change in State')
plt.legend()
plt.show()

for j in range(200):
    #k = random.randint(0, 3)
    n = random.randint(0, 499)
    y_stream0.append(Y[n][0])
    y_pred_stream0.append(Y_pred.T[n][0])
    y_stream1.append(Y[n][1])
    y_pred_stream1.append(Y_pred.T[n][1])
    y_stream2.append(Y[n][2])
    y_pred_stream2.append(Y_pred.T[n][2])
    y_stream3.append(Y[n][3])
    y_pred_stream3.append(Y_pred.T[n][3])

plt.scatter(y_stream1, y_pred_stream1, label='change in Cart Velocity')
plt.scatter(y_stream2, y_pred_stream2, label='change in Pole Angle')
plt.scatter(y_stream3, y_pred_stream3, label='change in Pole Velocity')
plt.scatter(y_stream0, y_pred_stream0, label='change in Cart Position')
plt.xlabel('True change in State')
plt.ylabel('Predicted change in State')
plt.legend()
plt.show()

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

for j in range(20):
    #k = random.randint(0, 3)
    n = random.randint(0, 499)
    y_stream0.append(Y[n][0])
    y_pred_stream0.append(Y_pred.T[n][0])
    y_stream1.append(Y[n][1])
    y_pred_stream1.append(Y_pred.T[n][1])
    y_stream2.append(Y[n][2])
    y_pred_stream2.append(Y_pred.T[n][2])
    y_stream3.append(Y[n][3])
    y_pred_stream3.append(Y_pred.T[n][3])


plt.plot(y_stream1, label='change in Cart Velocity')
plt.plot(y_stream2, label='change in Pole Angle')
plt.plot(y_stream3, label='change in Pole Velocity')
plt.plot(y_stream0, label='change in Cart Position')
plt.plot(y_pred_stream0, label='Predicted change in Cart Position', linestyle='dashed')
plt.plot(y_pred_stream1, label='Predicted change in Cart Velocity', linestyle='dashed')
plt.plot(y_pred_stream2, label='Predicted change in Pole Angle', linestyle='dashed')
plt.plot(y_pred_stream3, label='Predicted change in Pole Velocity', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('change in State')
plt.legend()
plt.show()
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
example_system = CartPole(visual=False)

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
#print(C)
Y_pred = C @ X.T

#print(Y_pred.T[3:7])
#print(Y[3:7])
# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
n = 100
X = np.empty((n, 4), float)
X_pred = np.empty((n, 4), float)

example_system = CartPole(visual=False)
cart_position = 0
cart_velocity = 1
pole_angle = 0
pole_velocity = 0
state = np.empty((1, 4), float)
state[0] = [cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity]
predicted_state = np.empty((1, 4), float)
predicted_state[0] = [cart_position, cart_velocity, remap_angle(pole_angle), pole_velocity]

cposition_stream = [0]*n
cvelocity_stream = [0]*n
pangle_stream = [0]*n
pvelocity_stream = [0]*n
pred_cposition_stream = [0]*n
pred_cvelocity_stream = [0]*n
pred_pangle_stream = [0]*n
pred_pvelocity_stream = [0]*n
print('starting predicted_state:', predicted_state)
example_system.setState(state[0])
for j in range(n):
    #print(predicted_state)
    print(j)
    X[j] = state[0]
    X_pred[j] = predicted_state[0]
    cposition_stream[j] = state[0][0]
    cvelocity_stream[j] = state[0][1]
    pangle_stream[j] = remap_angle(state[0][2])
    pvelocity_stream[j] = state[0][3]

    example_system.performAction()
    print((C @ predicted_state.T).T)
    Y_pred = (C @ predicted_state.T).T
    predicted_state[0][0] = predicted_state[0][0] + Y_pred[0][0]
    predicted_state[0][1] = predicted_state[0][1] + Y_pred[0][1]
    predicted_state[0][2] = predicted_state[0][2] + Y_pred[0][2]
    predicted_state[0][3] = predicted_state[0][3] + Y_pred[0][3]
    predicted_state[0][2] = remap_angle(predicted_state[0][2])
    pred_cposition_stream[j] = predicted_state[0][0]
    pred_cvelocity_stream[j] = predicted_state[0][1]
    pred_pangle_stream[j] = remap_angle(predicted_state[0][2])
    pred_pvelocity_stream[j] = predicted_state[0][3]
    state[0] = example_system.getState()
    
plt.plot(cvelocity_stream, label='Cart Velocity')
plt.plot(pred_cvelocity_stream, label='Predicted Cart Velocity', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Cart Velocity') 
plt.legend()
plt.show()
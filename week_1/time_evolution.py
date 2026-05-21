import random
import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole

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
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-10, 10)
    pole_angle = random.uniform(-np.pi, np.pi)
    pole_velocity = random.uniform(-15, 15)
    state = [[cart_position, cart_velocity, pole_angle, pole_velocity]]
    X = np.append(X, state, axis=0)
    x_cart_position.append(state[0][0])
    x_cart_velocity.append(state[0][1])
    x_pole_angle.append(state[0][2])
    x_pole_velocity.append(state[0][3])

    example_system.setState(state[0])
    example_system.performAction()

    current_state = [example_system.getState()]
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
# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
X = np.empty((0, 4), float)
X_pred = np.empty((0, 4), float)

example_system = CartPole(visual=False)
cart_position = random.uniform(-2.5, 2.5)
cart_velocity = random.uniform(-10, 10)
pole_angle = random.uniform(-np.pi, np.pi)
pole_velocity = random.uniform(-15, 15)
state = np.empty((0, 4), float)
state = np.append(state, [[cart_position, cart_velocity, pole_angle, pole_velocity]], axis=0)
predicted_state = np.empty((0, 4), float)
predicted_state = np.append(predicted_state, [[cart_position, cart_velocity, pole_angle, pole_velocity]], axis=0)

cposition_stream = []
cvelocity_stream = []
pangle_stream = []
pvelocity_stream = []
pred_cposition_stream = []
pred_cvelocity_stream = []
pred_pangle_stream = []
pred_pvelocity_stream = []

for j in range(50):
    print(j)
    X = np.append(X, state, axis=0)
    X_pred = np.append(X_pred, predicted_state, axis=0)
    cposition_stream.append(state[0][0])
    cvelocity_stream.append(state[0][1])
    pangle_stream.append(state[0][2])
    pvelocity_stream.append(state[0][3])

    example_system.setState(state[0])
    example_system.performAction()
    #predicted_state = [(C @ predicted_state.T).flatten()]
    predicted_state = (C @ predicted_state.T).T
    pred_cposition_stream.append(predicted_state[0][0])
    pred_cvelocity_stream.append(predicted_state[0][1])
    pred_pangle_stream.append(predicted_state[0][2])
    pred_pvelocity_stream.append(predicted_state[0][3])
    state = [example_system.getState()]
    
plt.plot(cposition_stream, label='Cart Position')
plt.plot(pred_cposition_stream, label='Predicted Cart Position', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Cart Position') 
plt.legend()
plt.show()
import random
import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole

# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
X = []
x_cart_position = []
x_cart_velocity = []
x_pole_angle = []
x_pole_velocity = []
Y = []
d_cart_position = []
d_cart_velocity = []
d_pole_angle = []
d_pole_velocity = []

for i in range(500):
    example_system = CartPole(visual=False)
    cart_position = random.uniform(-2.5, 2.5)
    cart_velocity = random.uniform(-10, 10)
    pole_angle = -1
    pole_velocity = random.uniform(-15, 15)

    state = [cart_position, cart_velocity, pole_angle, pole_velocity]
    X.append(state)
    x_cart_position.append(state[0])
    x_cart_velocity.append(state[1])
    x_pole_angle.append(state[2])
    x_pole_velocity.append(state[3])

    example_system.setState(state)
    example_system.performAction()

    current_state = example_system.getState()
    Y.append(current_state - state)
    d_cart_position.append(current_state[0] - state[0])
    d_cart_velocity.append(current_state[1] - state[1])
    d_pole_angle.append(current_state[2] - state[2])
    d_pole_velocity.append(current_state[3] - state[3])
print(X)
print(Y) 
plt.figure()   
plt.tricontourf(x_pole_velocity, x_cart_velocity, d_cart_position, levels=50, cmap='viridis')
plt.colorbar(label='Change in Cart Position')
#plt.scatter(x_pole_velocity, d_cart_position, label='change in cart position vs pole velocity at t')
plt.xlabel('Pole Velocity')

plt.ylabel('Cart Velocity')
plt.show()
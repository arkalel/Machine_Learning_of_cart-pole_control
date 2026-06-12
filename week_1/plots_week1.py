import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole

# visual=True turns on animation (don’t use this in other sections!)
#example_system = CartPole(visual=True)
example_system = CartPole(visual=False)

cart_position = 0.0
cart_velocity = 0
pole_angle = 0
pole_velocity = 0
cvelocity = [cart_velocity]
pvelocity = [pole_velocity]
clocation = [cart_position]
plocation = [pole_angle]
time = [0.0]
dt = 0.1

state = [cart_position, cart_velocity, pole_angle, pole_velocity]
example_system.setState(state)
for x in range(500):
    example_system.performAction(-100)
    time.append(x*dt)
    current_state = example_system.getState()
    #print(current_state)
    clocation.append(current_state[0]) 
    cvelocity.append(current_state[1])
    plocation.append(current_state[2])
    pvelocity.append(current_state[3])
print(len(clocation))   
print(len(time))
plt.figure()  
#plt.plot(plocation, cvelocity) 

#plt.plot(time, plocation, label = 'Pole Location')
plt.plot(time, cvelocity, label = 'Cart Velocity')
#plt.plot(time, pvelocity, label = 'Pole Velocity')
#plt.ylabel('State')
#plt.legend()
#plt.xlabel('Time')
plt.ylabel('Cart Velocity')
plt.xlabel('time')
plt.show()
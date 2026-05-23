import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole
# visual=True turns on animation (don’t use this in other sections!)
example_system = CartPole(visual=True)
cart_position = 0.0
cart_velocity = 1
pole_angle = 0
pole_velocity = 0
#time = [0.0]
#dt = 0.1

state = [cart_position, cart_velocity, pole_angle, pole_velocity]
example_system.setState(state)
for x in range(50):
    example_system.performAction()
    #time.append(x*dt)
    #current_state = example_system.getState()
    #print(current_state)
    #clocation.append(current_state[0]) 
    #cvelocity.append(current_state[1])
    #plocation.append(current_state[2])
    #pvelocity.append(current_state[3])
#print(len(clocation))   
#print(len(time))
#plt.figure()   
#plt.plot(time, clocation)
#plt.ylabel('Cart Location')
#plt.xlabel('Time')
#plt.show()
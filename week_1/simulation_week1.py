import numpy as np
import jax
import matplotlib.pyplot as plt
from cartpole import CartPole, remap_angle
# visual=True turns on animation (don’t use this in other sections!)
example_system = CartPole(visual=True)
cart_position = 0.0
cart_velocity = 0
pole_angle = np.pi
pole_velocity = 0
#time = [0.0]
#dt = 0.1

current_state = [cart_position, cart_velocity, pole_angle, pole_velocity]
example_system.setState(current_state)
P = np.array([-44, -7, 8.7, 0.3])
for x in range(100):
    current_state[2] = remap_angle(current_state[2])
    action = P.dot(current_state)
    example_system.performAction(action)
    #time.append(x*dt)
    current_state = example_system.getState()
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
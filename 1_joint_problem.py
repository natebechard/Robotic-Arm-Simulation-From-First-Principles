import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp

#Simulation parameters:   ################################################

L = 1 #in meters, also known as L
B = 0.2 #rotational friction
Mass_1 = 1 # in kg, also known as M
motor_torque = 7 # in Nm, also known as Tq(t)
grav = 9.81 # m/s^2
simulationDuration = 20 # seconds

#initial conditions
initial_angle = -np.pi/2 - np.pi/8 #rad
initial_angular_velocity = 0 #rad/s
initial_values = [initial_angle, initial_angular_velocity]



def slopes(t, Y):     # Y is a vector of shape [x1, x2]
    return([Y[1], #x1'
          
          (motor_torque - Mass_1*grav*L*np.cos(Y[0])) / (Mass_1*L**2) - B*Y[1] #x2'

          ])
 # of the form: [x1', x2'], slopes are computed for Y values inputed into the function 

#finally, draw the vector field of the solution
plt.figure()
plt.ylabel('\u03B8\'(t)')
plt.xlabel('\u03B8(t)')

height = 10
width = 12
arrows_num = 3 #number of arrows in one unit
arrow_len = 1/(arrows_num*1.75)

for x in range(-3, int(width*arrows_num)):
    for v in range(int(-height*arrows_num), int(height*arrows_num)):
        P = slopes(0, [x/arrows_num,v/arrows_num])
        angle = np.arctan(P[1]/P[0])
        dx = np.cos(angle)*arrow_len
        dv = np.sin(angle)*arrow_len
        plt.arrow(x/arrows_num, v/arrows_num, dx, dv, color='blue', head_width = 1/(arrows_num*10))
#plt.plot([0,1], [0,1], '-or')

# Plotting parameters:    ###########################################################################
fig = plt.figure()
ax = plt.axes(xlim=(-L-3, L+3), ylim=(-L-3, L+3))
line, = ax.plot([], [], lw=2)
dots, = ax.plot([], [], marker="o")

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

def init():
    line.set_data([], [])
    dots.set_data([], [])
    return line, dots

ax.axis('equal')
txt = ax.text(0.5, 0.95, '', transform=ax.transAxes)
#####################################################################################################
#The ODE:
# y'' + By' + (siny)/L = Tq(t)/(M*L^2)                  y is the angle from the horizontal in radians
# first order decomposition: 
# y1' = y2
# y2' = -By2 - siny1/L + Tq(t)/(M*L^2)
T = np.linspace(0, simulationDuration, simulationDuration*100)

# now the RK78 solver
sol = solve_ivp(slopes, [0, simulationDuration], initial_values, t_eval=T, method = 'DOP853', rtol=1e-8, atol=1e-8)

for x in range(len(sol.y[0])):
    print("position: " + str(sol.y[0, x]) + "velocity: " + str(sol.y[1, x]))
# the RK78 solver outputs a vector of shape [X1, X2]


# now, to simulate it: 

x1 = np.cos(sol.y[0]) * L #convert angular solution coordinates to cartesian for animating
y1 = np.sin(sol.y[0]) * L 

print(len(x1)) # prints how many frames the simulation has

def animate(i):
    x = [0, x1[i]]  
    y = [0, y1[i]]
    line.set_data(x, y) # make a line between (0,0) and (x1[i], y1[i])
    dots.set_data(x, y) # make dots at (0,0) and (x1[i], y1[i])
    txt.set_text('Time = ' + '{:4.1f}'.format(i*(simulationDuration/len(x1)))+ 's' + "  Velocity: " + '{:4.1f}'.format(sol.y[1,i]) + " rad/s") # display time and velocity 
    return line, dots, txt

# call the animator.  blit=True means only re-draw the parts that have changed.s
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1), interval=10, blit=True)  # each frame iterates i by one. There is a 10ms delay between frames

plt.show()

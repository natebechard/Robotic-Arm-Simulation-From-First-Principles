import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp

#Simulation parameters:   ################################################

L = 5 #in meters, also known as L
B = 0.5 #rotational friction
Mass_1 = 1 # in kg, also known as M
motor_torque = 50 # in Nm, also known as Tq(t)
grav = 9.81 # m/s^2
simulationDuration = 100 # seconds

#initial conditions
initial_angle = 0 #rad
initial_angular_velocity = 0 #rad/s
initial_values_of_animation = [initial_angle, initial_angular_velocity]

alt_angles = [1, 0]
alt_velocities = [4, -2]

def slopes(t, Y):     # Y is a vector of shape [x1, x2]
    return([Y[1], #x1'
          
          (motor_torque - Mass_1*grav*L*np.cos(Y[0])) / (Mass_1*L**2) - B*Y[1] #x2'

          ])
 # of the form: [x1', x2'], slopes are computed for Y values inputed into the function 

#plot the phase portrait of the ODE ###########################################################################
plt.figure()
plt.ylabel('\u03B8\'(t)')
plt.xlabel('\u03B8(t)')

height = 10
width = 15
arrows_num = 2 #number of arrows in one unit
arrow_len = 1/(arrows_num*1.75)

for x in range(-3, int(width*arrows_num)):
    for v in range(int(-height*arrows_num), int(height*arrows_num)):
        P = slopes(0, [x/arrows_num,v/arrows_num])
        angle = np.arctan(P[1]/P[0])
        if v < 0: 
            dx = -np.cos(angle)*arrow_len
            dv = -np.sin(angle)*arrow_len
        else:
            dx = np.cos(angle)*arrow_len
            dv = np.sin(angle)*arrow_len

        plt.arrow(x/arrows_num, v/arrows_num, dx, dv, color='blue', head_width = 1/(arrows_num*10))
#plt.plot([0,1], [0,1], '-or')

######################################################################################################################################################



T = np.linspace(0, simulationDuration, simulationDuration*100)

# now the RK78 solver
sol = solve_ivp(slopes, [0, simulationDuration], initial_values_of_animation, t_eval=T, method = 'DOP853', rtol=1e-8, atol=1e-8)

#for x in range(len(sol.y[0])):
#    print("position: " + str(sol.y[0, x]) + "velocity: " + str(sol.y[1, x]))
# the RK78 solver outputs a vector of shape [X1, X2]


#add alternate IVPs to plot multiple lines on phase portrait
sol2 = solve_ivp(slopes, [0, simulationDuration], [alt_angles[0],alt_velocities[0]], t_eval=T, method = 'DOP853', rtol=1e-8, atol=1e-8)
sol3 = solve_ivp(slopes, [0, simulationDuration], [alt_angles[1],alt_velocities[1]], t_eval=T, method = 'DOP853', rtol=1e-8, atol=1e-8)

# Plotting IVP's on phase portrait:    ###########################################################################
num = 10000
X1, X2, Xa, Xb, Xi, Xii = np.zeros(num),np.zeros(num),np.zeros(num),np.zeros(num),np.zeros(num),np.zeros(num)
x = 0
while abs(sol.y[0,x])<width and abs(sol.y[1,x])<height and x < num-1 :
    X1[x]= sol.y[0,x]
    X2[x] = sol.y[1,x]
    x += 1



y = 0
while abs(sol2.y[0,y])<width and abs(sol2.y[1,y])<height and y < num-1 :
    Xa[y]= sol2.y[0,y]
    Xb[y] = sol2.y[1,y]
    y += 1

z = 0
while abs(sol3.y[0,z])<width and abs(sol3.y[1,z])<height and z < num-1 :
    Xi[z]= sol3.y[0,z]
    Xii[z] = sol3.y[1,z]
    z += 1
plt.plot(X1[0:x], X2[0:x])
plt.plot(Xa[0:y], Xb[0:y])
plt.plot(Xi[0:z], Xii[0:z])
plt.plot([initial_angle], [initial_angular_velocity], marker="o")
plt.plot([alt_angles[0]], [alt_velocities[0]], marker="o")
plt.plot([alt_angles[1]], [alt_velocities[1]], marker="o")

######################################################################################################################################################


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



# Simulate the IVP, and animate it ##############################################################

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

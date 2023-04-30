import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp

#Simulation parameters:   ################################################

sn = 4

Lengths = [2, 1, 0.75, 2] # lenghths (m)
Bs = [0.5, 0.5, 0.5, 0.25] # friction terms 
Masses = [3, 5, 1, 3] # mass terms (kg)
Torques = [300, 3, 50, 5] # torque terms (Nm)
Init_values = [0.1,0.3,0.1,0.3,0.1,0.3,0,0] # position1, velocity1, position2, velocity2, ....., positionn, velocityn

grav = 9.81
simulationDuration = 20 # seconds

# Plotting parameters:    ###########################################################################
fig = plt.figure()

limit = 0
for x in range(len(Lengths)):
    limit+= Lengths[x]
limit += 2

ax = plt.axes(xlim=(-limit, limit), ylim=(-limit, limit))
line, = ax.plot([], [], lw=2)
dots, = ax.plot([], [], marker="o")

plt.title("N joints simulation")

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

def init():
    line.set_data([], [])
    dots.set_data([], [])
    return line, dots

ax.axis('equal')

# Time box, from the old ESP project
time_template = 'Time = '
txt = ax.text(0.8, 0.8, '', transform=ax.transAxes)

#####################################################################################################
#The ODE:
# y'' + By' + (cosy)/L = Tq(t)/(M*L^2)                  y is the angle from the horizontal in radians
# first order decomposition: 
# y1' = y2
# y2' = -By2 - cosy1/L + Tq(t)/(M*L^2)

def slopes(t, Input):     # Y is a vector of shape [x1, x2,..., xn]
    output = np.zeros(sn*2)

    #computes the distance between joints in an arm
    for joint_computed in range(sn):  # iterate through the segments to compute the rotational inertia on each joint
        rotational_inertia = 0 #rotational inertia WRT the joint being computed
        net_torque_on_joint = 0 #net torque on the joint being computed
        for mass_computed in range((sn-1),(joint_computed-1),-1):
            lx, ly = 0, 0
            dist = 0
            for y in range(joint_computed, (mass_computed + 1)): #the vector sum 
                lx = lx + Lengths[y]*np.cos(Input[y*2])
                ly = ly + Lengths[y]*np.sin(Input[y*2])
                print("Lx: " + str(lx))
                print("Ly: " + str(ly))
            dist = np.sqrt(lx**2 + ly**2) #The distance from Joint A to B

           # print("distance from motor " + str(joint_computed) + " to mass " + str(mass_computed) + " = " + str(dist))+ "the mass here is: " + str(Masses[mass_computed])
            rotational_inertia += Masses[mass_computed] * dist**2 # Mass*radius^2 The rotational inertia of the point mass computed is added to the system
            net_torque_on_joint += Masses[mass_computed]*grav*dist*(lx/dist) # lx/dist = adj/hyp = cos
        #take the inertia and torques calculated, and use them to compute the ODE:
       # print(Input[2*joint_computed + 1])
       # print(((Torques[joint_computed] - net_torque_on_joint) / rotational_inertia)  - Bs[joint_computed]*Input[2*joint_computed + 1])
        output[joint_computed*2] = Input[2*joint_computed + 1]
        output[joint_computed*2 +1] = ((Torques[joint_computed] - net_torque_on_joint) / rotational_inertia)  - Bs[joint_computed]*Input[2*joint_computed + 1] #x2'

        print("rotational inertia wrt to motor #" + str(joint_computed) + " : " + str(rotational_inertia))

    return(output)





# of the form: [x1', x2',..., xn'], slopes are computed for Y values inputed into the function #flipped the sin for a cos here to rectify angle tourques

T = np.linspace(0, simulationDuration, simulationDuration*100)

# now the RK78 solver
sol = solve_ivp(slopes, [0, simulationDuration], Init_values, t_eval=T, method = 'DOP853', rtol=1e-8, atol=1e-8)

print(sol.y[0])
for x in range(len(sol.y[0])):
    print("position1: " + str(sol.y[0, x]) + "position2: " + str(sol.y[2, x]) + "position3: " + str(sol.y[4, x]))

#for x in range(len(sol.y[0])):
#    print("time " + str(0.01*x) + "s, Li: " + str(np.sqrt((L*np.cos(sol.y[0, x]) + L2*np.cos(sol.y[2, x]))**2  +  (L*np.sin(sol.y[0, x]) + L2*np.sin(sol.y[2, x]))**2)) + " Velocity_of_motor_1: " + str(sol.y[1, x]))


# now, to animate it: 

#setup the arrays of coordinates for the animation
x_coords = np.zeros((sn+1, len(sol.y[0])))
y_coords = np.zeros((sn+1, len(sol.y[0])))

x = 0
y = 0

for joint_num in range(sn):
    x += Lengths[joint_num]*np.cos(sol.y[2*joint_num])
    y += Lengths[joint_num]*np.sin(sol.y[2*joint_num])
    x_coords[joint_num+1] = x
    y_coords[joint_num+1] = y

#animator call
def animate(i):
    
    line.set_data(x_coords[:, i], y_coords[:, i])
    dots.set_data(x_coords[:, i], y_coords[:, i])
    txt.set_text(time_template + '{:4.1f}'.format(i*(simulationDuration/len(sol.y[0]))) + 's')
    return line, dots, txt

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(sol.y[0]), interval=10, blit=True)  # each frame iterates i by one, In the arm model, we can make that 0.01 seconds or something like that #interval is the delay between frames in miliseconds
plt.show()

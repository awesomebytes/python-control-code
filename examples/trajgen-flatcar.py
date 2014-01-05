# trajgen-flatcar.py - differential flatness example (kinematic car)
# RMM, 18 Nov 2012
#
# This example shows how to compute a trajectory for a kinematic car model,
# using a sinusoidal family of basis functions.  The main purpose of this
# example is to show how to set up a differential flat system completely
# from scratch.

import types
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl                 # control system toolbox
import control.trajgen as tg           # trajectory generation toolbox

# Define the forward map for the kinematic car: states -> flat flag
def kincar_forward(self, x, u = np.zeros(2)):
    # Start with the easy calculations: position and velocity
    xpos = x[0]; ypos = x[1]; theta = x[2]
    xvel = u[0] * np.cos(theta)
    yvel = u[0] * np.sin(theta)

    # Next compute the speed and angular rate
    vel = u[0]
    omega = (vel / self.wheelbase) * np.tan(u[1])

    # For acceleration, assume vdot = 0 for simplicity (still invertible)
    xacc = -vel * omega * np.sin(theta)
    yacc =  vel * omega * np.cos(theta)

    # Return the flag
    return ((xpos, xvel, xacc), (ypos, yvel, yacc))

# Define the forward map for the kinematic car: flat flag -> states
def kincar_reverse(self, zflag):
    # Save the basic variables in a les cryptic form
    (xpos, xvel, xacc) = zflag[0]
    (ypos, yvel, yacc) = zflag[1]

    # Next figure out the angle and velocity
    theta = np.arctan2(yvel, xvel)
    vel = xvel * np.cos(theta) + yvel * np.sin(theta)

    # Solve the angular rate (see FBS, equation (7.24))
    omega = (yacc * np.cos(theta) - xacc * np.sin(theta)) / vel

    # And finally, the steering angle
    delta = np.arctan(omega * self.wheelbase / vel)

    # Now save the states and inputs
    x = (xpos, ypos, theta)
    u = (vel, delta)
    return x, u

# Define the flat system object
kincar = tg.FlatSystem(4, 2)
kincar.wheelbase = 1
kincar.forward = types.MethodType(kincar_forward, kincar)
kincar.reverse = types.MethodType(kincar_reverse, kincar)
        
# Define the basis functions to be used (not yet implemented)
class sinusoids(tg.BasisFamily):
    # psi_2k = sin(k omega t)
    # psi_2k+1 = cos(k omega t)
    def __init__(self, N, omega=2*np.pi):
        tg.BasisFamily.__init__(N)
        self.omega = omega

    def eval_deriv(self, i, j, t):
        # Figure out the frequency
        k = int(j/2); evenfcn = j - 2*k;
        if (evenfcn):
            return pow(omega, i) * sin(omega * k * t)

# Set the initial and final conditions
x0 = (0, 0, 0);
xf = (10, 1, 0);

# Find a trajectory
systraj = tg.point_to_point(kincar, x0, xf, 1)

# Plot the trajectory
t = np.linspace(0, 1, 100)
xd, ud = systraj.eval(t)

plt.figure(1); plt.clf()
plt.plot(t, xd[:,0], 'b-', t, xd[:,1], 'g-')
plt.legend(('x', 'xy'))
plt.show()

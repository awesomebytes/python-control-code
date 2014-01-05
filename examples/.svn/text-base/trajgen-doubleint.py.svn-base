# doubleint.py - double integrator example
# RMM, 10 Nov 2012
#
# This example shows how to compute a trajectory for a very simple double
# integrator system.  Mainly useful to show the simplest type of trajectory
# generation computation.

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl                 # control system toolbox
import control.trajgen as tg           # trajectory generation toolbox

# Define a double integrator system
sys1 = ctrl.tf2ss(ctrl.tf([1], [1, 0, 0]))
sysf = tg.LinearFlatSystem(sys1)

# Set the initial and final conditions
x0 = (0, 0);
xf = (1, 3);

# Find a trajectory
systraj = tg.point_to_point(sysf, x0, xf, 1)

# Plot the trajectory
t = np.linspace(0, 1, 100)
xd, ud = systraj.eval(t)

plt.figure(1); plt.clf()
plt.plot(t, xd[:,0], 'b-', t, xd[:,1], 'g-', t, ud[:,0], 'r--')
plt.legend(('x1', 'x2', 'u'))
plt.show()

# flatsys.py - trajectory generation for differentially flat systems
# RMM, 10 Nov 2012
#
# This file contains routines for computing trajectories for differentially
# flat nonlinear systems.  It is (very) loosely based on the NTG software
# package developed by Mark Milam and Kudah Mushambi, but rewritten from
# scratch in python.
#
# Copyright (c) 2012 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import numpy as np
import control
from control.trajgen import PolyFamily
from control.trajgen import SystemTrajectory

# Flat system class (for use as a base class)
class FlatSystem:
    """
    The FlatSystem class is used as a base class to describe differentially
    flat systems for trajectory generation.  The class must implement two
    functions:

    zflag = flatsys.foward(x, u)
        This function computes the flag (derivatives) of the flat output.
        The inputs to this function are the state 'x' and inputs 'u' (both
        1D arrays).  The output should be a 2D array with the first
        dimension equal to the number of system inputs and the second
        dimension of the length required to represent the full system
        dynamics (typically the number of states)

    x, u = flatsys.reverse(zflag)
        This function system state and inputs give the the flag
        (derivatives) of the flat output.  The input to this function is an
        2D array whose first dimension is equal to the number of system
        inputs and whose second dimension is of length required to represent
        the full system dynamics (typically the number of states).  The
        output is the state 'x' and inputs 'u' (both 1D arrays).
    """
    def __init__(self, states, inputs):
        # Save the number of inputs and outputs
        self.states = states
        self.inputs = inputs

# Solve a point to point trajectory generation problem for a linear system
def point_to_point(sys, x0, xf, Tf, T0 = 0, basis=None, cost=None):
    """
    Compute a trajectory between an initial condition and final condition

      traj = point_to_point(flatsys, x0, xf, Tf)

    Parameters
    ----------
    flatsys : FlatSystem object
        Description of the differentially flat system.  This object must
        define a function flatsys.forward() that takes the system state and
        produceds the flag of flat outputs and a system flatsys.reverse()
        that takes the flag of the flat output and prodes the state and
        input.

    x0, xf : 1D arrays
        Define the desired initial and final conditions for the system

    Tf : float
        The final time for the trajectory (corresponding to xf)

    T0 : float (optional)
        The initial time for the trajectory (corresponding to x0).  If not
        specified, its value is taken to be zero.

    basis : BasisFamily object (optional)
        The basis functions to use for generating the trajectory.  If not
        specified, the PolyFamily basis family will be used, with the minimal
        number of elements required to find a feasible trajectory (twice
        the number of system states)

    Returns
    -------
    traj : SystemTrajectory object
        The system trajectory is returned as an object that implements the
        eval() function, we can be used to compute the value of the state
        and input and a given time t.

    """
    #
    # Make sure the probelm is one that we can handle
    #
    #! TODO: put in tests for flat system input
    #! TODO: process initial and final conditions to allow x0 or (x0, u0)

    #
    # Determine the basis function set to use and make sure it is big enough
    #

    # If no basis set was specified, use a polynomial basis (poor choice...)
    if (basis is None): basis = PolyFamily(2*sys.states)
    
    # Make sure we have enough basis functions to solve the problem
    if (basis.N < 2*sys.states):
        raise ValueError("basis set is too small")

    #
    # Map the initial and final conditions to flat output conditions
    #
    # We need to compute the output "flag": [z(t), z'(t), z''(t), ...]
    # and then evaluate this at the initial and final condition.
    #
    #! TODO: should be able to represent flag variables as 1D arrays
    #! TODO: need inputs to fully define the flag
    zflag_T0 = sys.forward(x0)
    zflag_Tf = sys.forward(xf)

    #
    # Compute the matrix constraints for initial and final conditions
    #
    # This computation depends on the basis function we are using.  It
    # essentially amounts to evaluating the basis functions and their
    # derivatives at the initial and final conditions.

    # Start by creating an empty matrix that we can fill up
    M = np.zeros((2*sys.states, basis.N))

    # Now fill in the rows for the initial and final states
    for i in range(sys.states):
        for j in range(basis.N):
            M[i, j] = basis.eval_deriv(j, i, T0)
            M[sys.states + i, j] = basis.eval_deriv(j, i, Tf)

    #
    # Solve for the coefficients of the flat outputs
    #
    # At this point, we need to solve the equation M alpha = zflag, where M
    # is the matrix constrains for initial and final conditions and zflag =
    # [zflag_T0; zflag_tf].  Since everything is linear, just compute the
    # least squares solution for now.
    #
    #! TODO: need to allow cost and constraints...
    alpha = np.dot(np.linalg.pinv(M), np.vstack((zflag_T0, zflag_Tf)))

    #
    # Transform the trajectory from flat outputs to states and inputs
    #
    systraj = SystemTrajectory(sys.states, sys.inputs)
    systraj.system = sys
    systraj.basis = basis
    systraj.coeffs = alpha

    # Return a function that computes inputs and states as a function of time
    return systraj


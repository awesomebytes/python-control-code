# linflat.py - FlatSystem subclass for linear systems
# RMM, 10 November 2012
#
# This file defines a FlatSystem class for a linear system.
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
from control.trajgen.flatsys import FlatSystem

class LinearFlatSystem(FlatSystem):
    def __init__(self, sys):
        # Make sure we can handle the system
        if (not control.isctime(sys)):
            raise control.ControlNotImplemented(
                "requires continuous time, linear control system")
        elif (not control.issiso(sys)):
            raise control.ControlNotImplemented(
                "only single input, single output systems are supported")

        # Initialize the object and store system matrices
        FlatSystem.__init__(self, sys.states, sys.inputs)
        self.A = sys.A
        self.B = sys.B

        # Find the transformation to bring the system into reachable form
        zsys, Tr = control.reachable_form(sys)
        self.F = zsys.A[0,:]            # input function coeffs
        self.T = Tr                     # state space transformation
        self.Tinv = np.linalg.inv(Tr)   # computer inverse once
        
        # Compute the flat output variable z = C x
        Cfz = np.zeros(np.shape(sys.C)); Cfz[0, -1] = 1
        self.C = Cfz * Tr

    # Compute the flat flag from the state (and input)
    def forward(self, x, u=None):
        zflag = np.zeros((self.states, 1));
        H = self.C              # initial state transformation
        for i in range(self.states):
            zflag[i, 0] = H * np.matrix(x).T
            H = H * self.A      # derivative for next iteration
        return zflag

    # Compute state and input from flat flag
    def reverse(self, zflag):
        z = np.matrix(zflag[-2::-1]).T
        x = self.Tinv * z
        u = zflag[-1] - self.F * z
        return np.reshape(x, self.states), np.reshape(u, self.inputs)

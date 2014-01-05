# systraj.py - SystemTrajectory class
# RMM, 10 November 2012
#
# The SystemTrajetory class is used to store a feasible trajectory for
# the state and input of a (nonlinear) control system.
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

class SystemTrajectory:
    def __init__(self, states, inputs):
        self.states = states
        self.inputs = inputs

    # Evaluate the trajectory over a list of time points
    def eval(self, tlist):
        # Allocate space for the outputs
        xd = np.zeros((len(tlist), self.states))
        ud = np.zeros((len(tlist), self.inputs))

        # Go through each time point and compute xd and ud via flat variables
        for k in range(len(tlist)):
            zflag = np.zeros(self.states + self.inputs)
            for i in range(self.states + self.inputs):
                for j in range(self.basis.N):
                    #! TODO: rewrite eval_deriv to take in time vector
                    zflag[i] += self.coeffs[j] * \
                        self.basis.eval_deriv(j, i, tlist[k])

            # Now copy the states and inputs
            xd[k,:], ud[k,:] = self.system.reverse(zflag)

        return xd, ud


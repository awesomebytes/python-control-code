"""bdalg.py

This file contains some standard block diagram algebra.

Routines in this module:

append
series
parallel
negate
feedback
connect

"""

"""Copyright (c) 2010 by California Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Richard M. Murray
Date: 24 May 09
Revised: Kevin K. Chen, Dec 10

$Id: bdalg.py 276 2013-06-09 17:07:36Z murrayrm $

"""

import scipy as sp
import control.xferfcn as tf
import control.statesp as ss

def series(sys1, sys2):
    """Return the series connection sys2 * sys1 for --> sys1 --> sys2 -->.

    Parameters
    ----------
    sys1: scalar, StateSpace, or TransferFunction
    sys2: scalar, StateSpace, or TransferFunction

    Returns
    -------
    out: scalar, StateSpace, or TransferFunction

    Raises
    ------
    ValueError
        if `sys2.inputs` does not equal `sys1.outputs`
        if `sys1.dt` is not compatible with `sys2.dt`

    See Also
    --------
    parallel
    feedback

    Notes
    -----
    This function is a wrapper for the __mul__ function in the StateSpace and
    TransferFunction classes.  The output type is usually the type of `sys2`.
    If `sys2` is a scalar, then the output type is the type of `sys1`.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> sys3 = series(sys1, sys2) # Same as sys3 = sys2 * sys1.

    """
    
    return sys2 * sys1

def parallel(sys1, sys2):
    """
    Return the parallel connection sys1 + sys2.

    Parameters
    ----------
    sys1: scalar, StateSpace, or TransferFunction
    sys2: scalar, StateSpace, or TransferFunction

    Returns
    -------
    out: scalar, StateSpace, or TransferFunction

    Raises
    ------
    ValueError
        if `sys1` and `sys2` do not have the same numbers of inputs and outputs
            
    See Also
    --------
    series
    feedback
    
    Notes
    -----
    This function is a wrapper for the __add__ function in the
    StateSpace and TransferFunction classes.  The output type is usually
    the type of `sys1`.  If `sys1` is a scalar, then the output type is
    the type of `sys2`.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> sys3 = parallel(sys1, sys2) # Same as sys3 = sys1 + sys2.

    """
    
    return sys1 + sys2

def negate(sys):
    """
    Return the negative of a system.

    Parameters
    ----------
    sys: StateSpace or TransferFunction

    Returns
    -------
    out: StateSpace or TransferFunction

    Notes
    -----
    This function is a wrapper for the __neg__ function in the StateSpace and
    TransferFunction classes.  The output type is the same as the input type.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> sys2 = negate(sys1) # Same as sys2 = -sys1.

    """
    
    return -sys;

#! TODO: expand to allow sys2 default to work in MIMO case?
def feedback(sys1, sys2=1, sign=-1):
    """
    Feedback interconnection between two I/O systems.

    Parameters
    ----------
    sys1: scalar, StateSpace, or TransferFunction
        The primary plant.
    sys2: scalar, StateSpace, or TransferFunction
        The feedback plant (often a feedback controller).
    sign: scalar 
        The sign of feedback.  `sign` = -1 indicates negative feedback, and
        `sign` = 1 indicates positive feedback.  `sign` is an optional
        argument; it assumes a value of -1 if not specified.

    Returns
    -------
    out: StateSpace or TransferFunction

    Raises
    ------
    ValueError
        if `sys1` does not have as many inputs as `sys2` has outputs, or if
        `sys2` does not have as many inputs as `sys1` has outputs
    NotImplementedError
        if an attempt is made to perform a feedback on a MIMO TransferFunction
        object

    See Also
    --------
    series
    parallel

    Notes
    -----
    This function is a wrapper for the feedback function in the StateSpace and
    TransferFunction classes.  It calls TransferFunction.feedback if `sys1` is a
    TransferFunction object, and StateSpace.feedback if `sys1` is a StateSpace
    object.  If `sys1` is a scalar, then it is converted to `sys2`'s type, and
    the corresponding feedback function is used.  If `sys1` and `sys2` are both
    scalars, then TransferFunction.feedback is used.
  
    """

    # Check for correct input types.
    if not isinstance(sys1, (int, float, complex, tf.TransferFunction,
        ss.StateSpace)):
        raise TypeError("sys1 must be a TransferFunction or StateSpace object, \
or a scalar.")
    if not isinstance(sys2, (int, float, complex, tf.TransferFunction,
        ss.StateSpace)):
        raise TypeError("sys2 must be a TransferFunction or StateSpace object, \
or a scalar.")

    # If sys1 is a scalar, convert it to the appropriate LTI type so that we can
    # its feedback member function.
    if isinstance(sys1, (int, float, complex)):
        if isinstance(sys2, tf.TransferFunction):
            sys1 = tf._convertToTransferFunction(sys1)
        elif isinstance(sys2, ss.StateSpace):
            sys1 = ss._convertToStateSpace(sys1)
        else: # sys2 is a scalar.
            sys1 = tf._convertToTransferFunction(sys1)
            sys2 = tf._convertToTransferFunction(sys2)

    return sys1.feedback(sys2, sign)

def append(*sys):
    '''
    Group models by appending their inputs and outputs

    Forms an augmented system model, and appends the inputs and
    outputs together. The system type will be the type of the first
    system given; if you mix state-space systems and gain matrices,
    make sure the gain matrices are not first.

    Parameters.
    -----------
    sys1, sys2, ... sysn: StateSpace or Transferfunction
        LTI systems to combine

        
    Returns
    -------
    sys: LTI system
        Combined LTI system, with input/output vectors consisting of all 
        input/output vectors appended
        
    Examples
    --------
    >>> sys1 = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> sys2 = ss("-1.", "1.", "1.", "0.")
    >>> sys = append(sys1, sys2)
    
    .. todo::
        also implement for transfer function, zpk, etc.
    '''
    s1 = sys[0]
    for s in sys[1:]:
        s1 = s1.append(s)
    return s1

def connect(sys, Q, inputv, outputv):
    '''
    Index-base interconnection of system

    The system sys is a system typically constructed with append, with
    multiple inputs and outputs. The inputs and outputs are connected
    according to the interconnection matrix Q, and then the final
    inputs and outputs are trimmed according to the inputs and outputs
    listed in inputv and outputv.

    Note: to have this work, inputs and outputs start counting at 1!!!!

    Parameters.
    -----------
    sys: StateSpace Transferfunction
        System to be connected
    Q: 2d array
        Interconnection matrix. First column gives the input to be connected
        second column gives the output to be fed into this input. Negative
        values for the second column mean the feedback is negative, 0 means
        no connection is made
    inputv: 1d array
        list of final external inputs
    outputv: 1d array
        list of final external outputs

    Returns
    -------
    sys: LTI system
        Connected and trimmed LTI system

    Examples
    --------
    >>> sys1 = ss("1. -2; 3. -4", "5.; 7", "6, 8", "9.")
    >>> sys2 = ss("-1.", "1.", "1.", "0.")
    >>> sys = append(sys1, sys2)
    >>> Q = sp.mat([ [ 1, 2], [2, -1] ]) # basically feedback, output 2 in 1
    >>> sysc = connect(sys, Q, [2], [1, 2])
    '''
    # first connect
    K = sp.zeros( (sys.inputs, sys.outputs) )
    for r in sp.array(Q).astype(int):
        inp = r[0]-1
        for outp in r[1:]:
            if outp > 0 and outp <= sys.outputs:
                K[inp,outp-1] = 1.
            elif outp < 0 and -outp >= -sys.outputs:
                K[inp,-outp-1] = -1.
    sys = sys.feedback(sp.matrix(K), sign=1)
    
    # now trim
    Ytrim = sp.zeros( (len(outputv), sys.outputs) )
    Utrim = sp.zeros( (sys.inputs, len(inputv)) )
    for i,u in enumerate(inputv):
        Utrim[u-1,i] = 1.
    for i,y in enumerate(outputv):
        Ytrim[i,y-1] = 1.
    return sp.matrix(Ytrim)*sys*sp.matrix(Utrim)  

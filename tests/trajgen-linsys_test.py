#!/usr/bin/env python
#
# linsys_test.py - unit tests for linear systems trajectory generation
# RMM, 11 Nov 2012
#
# This test suite creates a variety of linear systems and then generates
# trajectories and verifies that they satisfy the expected conditions.

from __future__ import print_function
import unittest
import numpy as np
import control as ctrl
import control.matlab as matlab
import control.trajgen as tg

class TestLinsys(unittest.TestCase):
    def setUp(self):
        self.maxStates = 5     # maximum number of states to try
        self.numTests = 4      # number of tests per system size
        self.debug = False     # turn on debugging output

    def test_point_to_point(self):
        # Machine precision for floats.
        eps = np.finfo(float).eps

        for states in range(1, self.maxStates):
            # Start with a random system
            linsys = matlab.rss(states, 1, 1)

            # Make sure the system is not degenerate
            Cmat = ctrl.ctrb(linsys.A, linsys.B)
            if (np.linalg.matrix_rank(Cmat) != states):
                if (self.debug):
                    print("  skipping (not reachable)")
                    continue

            if (self.debug): print(linsys)

            # Create a flat system representation
            flatsys = tg.LinearFlatSystem(linsys)

            # Generate several different initial and final conditions
            for i in range(self.numTests):
                x0 = np.random.rand(linsys.states)
                xf = np.random.rand(linsys.states)
                Tf = np.random.randn()

                # Generate a trajectory from start to stop
                systraj = tg.point_to_point(flatsys, x0, xf, Tf)
                xd, ud = systraj.eval((0,Tf))
                np.testing.assert_array_almost_equal(x0, xd[0,:], decimal=4)
                np.testing.assert_array_almost_equal(xf, xd[1,:], decimal=4)
        
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestLinsys)

if __name__ == '__main__':
    unittest.main()

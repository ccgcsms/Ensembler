import unittest
import numpy as np
from Ensembler.src import potentials as pot

class test_Integrators(unittest.TestCase):
    def test_Monte_Carlo_Integrator(self):
        expected_norm_dist = [1]

        s = 1
        Eoffs = [0, 0]
        V_is = [pot.OneD.harmonicOsc1D(x_shift=-10), pot.OneD.harmonicOsc1D(x_shift=10)]
        eds_pot = pot.ND.envelopedPotential(V_is=V_is, s=s, Eoff_i=Eoffs)

        positions = list(range(-100,100))
        energies = eds_pot.ene(positions)

        self.assert_(any([ex==ene for ex, ene in zip(expected, energies)]))

if __name__ == '__main__':
    unittest.main()
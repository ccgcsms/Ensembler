import unittest
from Ensembler.src import system, conveyorBelt as ensemble, integrator as integ
from Ensembler.src import potentials as potent

class testEnsemble(unittest.TestCase):
    def testEnsemble(self):
<<<<<<< HEAD
        ens = ensemble.ConveyorBeltEnsemble(0.0, 1)
=======
        ens = ensemble.ConveyorBelt(0.0, 1)
>>>>>>> a8de56b052c9ad75cc4dfd9a40065c547443ee0c
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystem(self):
        integrator = integ.metropolisMonteCarloIntegrator()
        ha = potent.OneD.harmonicOsc(x_shift=-5)
        hb = potent.OneD.harmonicOsc(x_shift=5)
        pot = potent.OneD.linCoupledHosc(ha=ha, hb=hb)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
<<<<<<< HEAD
        ens = ensemble.ConveyorBeltEnsemble(0.0, 1, system=sys)
=======
        ens = ensemble.ConveyorBelt(0.0, 1, system=sys)
>>>>>>> a8de56b052c9ad75cc4dfd9a40065c547443ee0c

        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystemShift(self):
        integrator = integ.metropolisMonteCarloIntegrator()
        ha = potent.OneD.harmonicOsc(x_shift=-5)
        hb = potent.OneD.harmonicOsc(x_shift=5)
        lam = 0.5
        pot = potent.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
<<<<<<< HEAD
        ens = ensemble.ConveyorBeltEnsemble(0.0, 1, system=sys)
=======
        ens = ensemble.ConveyorBelt(0.0, 1, system=sys)
>>>>>>> a8de56b052c9ad75cc4dfd9a40065c547443ee0c
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testTraj(self):
        integrator = integ.metropolisMonteCarloIntegrator()
        ha = potent.OneD.harmonicOsc(x_shift=-5)
        hb = potent.OneD.harmonicOsc(x_shift=5)
        lam = 0.5
        pot = potent.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        sys = system.perturbedSystem(temperature=300.0, potential=pot, integrator=integrator)
<<<<<<< HEAD
        ens = ensemble.ConveyorBeltEnsemble(0.0, 1, system=sys)
=======
        ens = ensemble.ConveyorBelt(0.0, 1, system=sys)
>>>>>>> a8de56b052c9ad75cc4dfd9a40065c547443ee0c

        print(ensemble.calc_traj(steps=10, ens=ens))
        ens = ensemble.ConveyorBeltEnsemble(0.0, 8, system=sys)

        #ensemble.calc_traj_file(steps=100, ens=ens)
        #import os
        #os.remove(os.getcwd()+"/traj_*.dat")

if __name__ == '__main__':
    unittest.main()

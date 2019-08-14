import unittest
from Ensembler.src import system, conveyorBelt as ensemble

class testEnsemble(unittest.TestCase):
    def testEnsemble(self):
        ens = ensemble.Ensembler(0.0, 1)
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystem(self):
        ens = ensemble.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystemShift(self):
        ens = ensemble.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testTraj(self):
        ens = ensemble.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        print(ensemble.calc_traj(steps=10, ens=ens))
        ens = ensemble.Ensembler(0.0, 8, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        Ensemble.calc_traj_file(steps=100, ens=ens)


if __name__ == '__main__':
    unittest.main()

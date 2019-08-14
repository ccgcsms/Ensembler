import unittest
from src import system, Ensembler


class testEnsemble(unittest.TestCase):
    def testEnsemble(self):
        ens = Ensembler.Ensembler(0.0, 1)
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystem(self):
        ens = Ensembler.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testEnsembleSystemShift(self):
        ens = Ensembler.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        ens.calc_ene()
        ens.propagate()
        ens.calc_ene()
        ens.print_systems()

    def testTraj(self):
        ens = Ensembler.Ensembler(0.0, 1, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        print(Ensembler.calc_traj(steps=10, ens=ens))
        ens = Ensembler.Ensembler(0.0, 8, system=system.system(temp=300.0,
                                                                     fc=1.0,
                                                                     lam=0.5,
                                                                     alpha=10.0,
                                                                     integrator='sd'))
        Ensembler.calc_traj_file(steps=100, ens=ens)


if __name__ == '__main__':
    unittest.main()

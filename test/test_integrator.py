import unittest
import numpy as np
from Ensembler.src import integrator as integ

#class test_Integrators(unittest.TestCase):
class test_MonteCarlo_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.monteCarloIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_MetropolisMonteCarlo_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.metropolisMonteCarloIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_verlocityVerletIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.velocityVerletIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_positionVerletIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.positionVerletIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

class test_leapFrogIntegrator_Integrator(unittest.TestCase):

    def test_constructor(self):
        integrator = integ.leapFrogIntegrator()

    def test_step(self):
        pass

    def test_integrate(self):
        pass

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import numbers

from Ensembler.src.potentials import OneD as potential1D
from Ensembler.src import potentials

from Ensembler.src import system, integrator
from Ensembler.src import dataStructure as data

class test_System(unittest.TestCase):
    def test_system_constructor(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005000000000000001, totPotEnergy=0.005000000000000001, totKinEnergy=0,
                                         dhdpos=None, velocity=None)


        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        curState = sys.getCurrentState()

        #check attributes
        self.assertEqual(pot.nDim, sys.nDim, msg="Dimensionality was not the same for system and potential!")
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")

        #check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The initialised totEnergy is not correct!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The initialised totPotEnergy is not correct!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The initialised totKinEnergy is not correct!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The initialised velocity is not correct!")

    def test_append_state(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        expected_state = data.basicState(position=newPosition, temperature=temperature,
                                         totEnergy=62.5, totPotEnergy=50.0, totKinEnergy=12.5,
                                         dhdpos=3, velocity=newVelocity)

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)


        sys.append_state(newPosition= newPosition,newVelocity= newVelocity, newForces=newForces)
        curState = sys.getCurrentState()

        #check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The initialised totEnergy is not correct!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The initialised totPotEnergy is not correct!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The initialised totKinEnergy is not correct!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The initialised velocity is not correct!")

    def test_revertStep(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        newPosition2 = 13
        newVelocity2 = -4
        newForces2 = 8

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)

        sys.append_state(newPosition=newPosition, newVelocity=newVelocity, newForces=newForces)
        expected_state = sys.getCurrentState()
        sys.append_state(newPosition=newPosition2, newVelocity=newVelocity2, newForces=newForces2)
        not_expected_state = sys.getCurrentState()
        sys.revertStep()
        curState = sys.getCurrentState()

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The current Position is not equal to the one two steps before!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The current temperature is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The current totEnergy is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The current totPotEnergy is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The current totKinEnergy is not equal to the one two steps before!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The current dhdpos is not equal to the one two steps before!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The current velocity is not equal to the one two steps before!")

        #check that middle step is not sames
        self.assertNotEqual(curState.position, not_expected_state.position, msg="The not expected Position equals the current one!")
        self.assertEqual(curState.temperature, not_expected_state.temperature, msg="The not expected temperature equals the current one")
        self.assertNotAlmostEqual(curState.totEnergy, not_expected_state.totEnergy, msg="The not expected totEnergy equals the current one")
        self.assertNotAlmostEqual(curState.totPotEnergy, not_expected_state.totPotEnergy, msg="The not expected totPotEnergy equals the current one")
        self.assertNotAlmostEqual(curState.totKinEnergy, not_expected_state.totKinEnergy, msg="The not expected totKinEnergy equals the current one")
        self.assertNotEqual(curState.dhdpos, not_expected_state.dhdpos, msg="The not expected dhdpos, equals the current one")
        self.assertNotEqual(curState.velocity, not_expected_state.velocity, msg="The not expected velocity equals the current one")

    def test_propergate(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005000000000000001, totPotEnergy=0.005000000000000001, totKinEnergy=0,
                                         dhdpos=None, velocity=None)


        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        initialState = sys.getCurrentState()
        sys.propagate()

        #check that middle step is not sames
        self.assertNotEqual(sys._currentPosition, initialState.position, msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertEqual(sys._currentTemperature, initialState.temperature, msg="The initialState does not equal the currentState after propergating in attribute: temperature!")
        self.assertAlmostEqual(sys._currentTotPot, initialState.totPotEnergy, msg="The initialState  does not equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(sys._currentTotKin, initialState.totKinEnergy, msg="The initialState  does not equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertNotEqual(sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity, msg="The initialState does not equal the currentState after propergating in attribute: velocity!")

    def test_simulate(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        steps = 100

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        init_state = sys.getCurrentState()
        sys.simulate(steps=steps, initSystem=False, withdrawTraj=True)  #withdrawTraj is needed in the context because of the interaction between different Tests
        trajectory = sys.getTrajectory()

        old_frame = trajectory[0]

        #Check that the first frame is the initial state!
        self.assertEqual(init_state.position, old_frame.position, msg="The initial state does not equal the frame 0 after propergating in attribute: Position!")
        self.assertEqual(init_state.temperature, old_frame.temperature,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: temperature!")
        self.assertAlmostEqual(init_state.totPotEnergy, old_frame.totPotEnergy,
                                  msg="The initial state does not equal the frame 0 after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(init_state.totKinEnergy, old_frame.totKinEnergy,
                                  msg="The initial state does not equal the frame 0 after propergating in attribute: totKinEnergy!")
        self.assertEqual(init_state.dhdpos, old_frame.dhdpos, msg="The initial state does not equal the frame 0 after propergating in attribute: dhdpos!")
        self.assertEqual(init_state.velocity, old_frame.velocity,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: velocity!")

        #check that the frames are all different from each other.
        for ind, frame in enumerate(trajectory[1:]):
            #check that middle step is not sames
            self.assertNotEqual(old_frame.position, frame.position, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: Position!")
            self.assertEqual(old_frame.temperature, frame.temperature, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: temperature!")    #due to integrator
            self.assertNotAlmostEqual(old_frame.totPotEnergy, frame.totPotEnergy, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: totPotEnergy!")
            self.assertAlmostEqual(old_frame.totKinEnergy, frame.totKinEnergy, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: totKinEnergy!")    #due to integrator
            self.assertNotEqual(old_frame.dhdpos, frame.dhdpos, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: dhdpos!")
            self.assertEqual(old_frame.velocity, frame.velocity, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: velocity!")     #due to integrator
            old_frame = frame

    def test_applyConditions(self):
        """
        NOT IMPLEMENTED!
        """
        pass

    def test_initVel(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]

        newPosition = 10
        newVelocity = -5
        newForces = 3

        expected_state = data.basicState(position=newPosition, temperature=temperature,
                                         totEnergy=62.5, totPotEnergy=50.0, totKinEnergy=12.5,
                                         dhdpos=3, velocity=newVelocity)

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        sys.initVel()

        cur_velocity = sys._currentVelocities

        self.assertEqual(type(cur_velocity), type(np.array([1])),msg="Velocity has not the correcttype!")
        self.assertTrue(all(map(lambda vel: isinstance(vel, numbers.Number),cur_velocity)),msg="Velocity has not the correcttype!")

    def test_updateTemp(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def test_updateEne(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005000000000000001, totPotEnergy=0.005000000000000001, totKinEnergy=0,
                                         dhdpos=None, velocity=None)


        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        initialState = sys.getCurrentState()
        sys.propagate()
        sys.updateEne()

        #check that middle step is not sames
        self.assertNotEqual(sys._currentPosition, initialState.position, msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertEqual(sys._currentTemperature, initialState.temperature, msg="The initialState does not equal the currentState after propergating in attribute: temperature!")
        self.assertNotAlmostEqual(sys._currentTotPot, initialState.totPotEnergy, msg="The initialState  does equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(sys._currentTotKin, initialState.totKinEnergy, msg="The initialState  does equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertNotEqual(sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity, msg="The initialState does not equal the currentState after propergating in attribute: velocity!")

    def test_totPot(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005000000000000001, totPotEnergy=0.005000000000000001, totKinEnergy=0,
                                         dhdpos=None, velocity=None)

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        self.assertAlmostEqual(sys.totPot(), 0.5, msg="The initialised totPotEnergy is not correct!")

    def test_totKin(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005000000000000001, totPotEnergy=0.005000000000000001, totKinEnergy=0,
                                         dhdpos=None, velocity=None)

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        self.assertAlmostEqual(sys.totKin(), 0.0, msg="The initialised totPotEnergy is not correct!")

        newPosition = 10
        newVelocity = -5
        newForces = 3
        sys.append_state(newPosition= newPosition,newVelocity= newVelocity, newForces=newForces)
        self.assertAlmostEqual(sys.totKin(), 12.5, msg="The initialised totPotEnergy is not correct!")

    def test_setTemperature(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        temperature2 = 600
        position = [0.1]
        mass = [1]

        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        sys._currentVelocities = 100
        sys.updateCurrentState()
        initialState = sys.getCurrentState()
        sys.set_Temperature(temperature2)

        # check that middle step is not sames
        self.assertEqual(sys._currentPosition, initialState.position,
                            msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertNotEqual(sys._currentTemperature, initialState.temperature,
                         msg="The initialState does equal the currentState after propergating in attribute: temperature!")
        self.assertAlmostEqual(sys._currentTotPot, initialState.totPotEnergy,
                                  msg="The initialState  does equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertNotAlmostEqual(sys._currentTotKin, initialState.totKinEnergy,
                               msg="The initialState  does not equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertEqual(sys._currentForce, initialState.dhdpos,
                            msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity,
                         msg="The initialState does equal the currentState after propergating in attribute: velocity!")

    def test_get_Pot(self):
        integ = integrator.monteCarloIntegrator()
        pot = potential1D.harmonicOsc()
        conditions = []
        temperature = 300
        position = [0.1]
        mass = [1]
        expected_state = data.basicState(position=position, temperature=temperature,
                                         totEnergy=0.005, totPotEnergy=0.005, totKinEnergy=0,
                                         dhdpos=None, velocity=None)


        sys = system.system(potential=pot, integrator=integ, position=position, temperature=temperature)
        self.assertEqual(0.005000000000000001, sys.getPot()[0], msg="Could not get the correct Pot Energy!")

class test_perturbedSystem(unittest.TestCase):
    def test_system_constructor(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam=0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=1.0)
        conditions = []
        temperature = 300
        position = [0]
        mass = [1]
        expected_state = data.lambdaState(position=position, temperature=temperature, lam=0.0,
                                         totEnergy=12.5, totPotEnergy=12.5, totKinEnergy=0,
                                         dhdpos=None, velocity=None,
                                          dhdlam=None)


        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        curState = sys.getCurrentState()

        #check attributes
        self.assertEqual(pot.nDim, sys.nDim, msg="Dimensionality was not the same for system and potential!")
        self.assertEqual([], sys.conditions, msg="Conditions were not empty!")

        #check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The initialised totEnergy is not correct!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The initialised totPotEnergy is not correct!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The initialised totKinEnergy is not correct!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The initialised velocity is not correct!")
        self.assertEqual(curState.lam, expected_state.lam, msg="The initialised lam is not correct!")
        self.assertEqual(curState.dhdlam, expected_state.dhdlam, msg="The initialised dHdlam is not correct!")

    def test_append_state(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        temperature = 300
        position = [0]

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1.0

        expected_state = data.lambdaState(position=newPosition, temperature=temperature, lam=newLam,
                                          totEnergy=125.0, totPotEnergy=112.5, totKinEnergy=12.5,
                                          dhdpos=newForces, velocity=newVelocity,
                                          dhdlam=None)

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature)

        sys.append_state(newPosition=newPosition, newVelocity=newVelocity, newForces=newForces, newLam=newLam)
        curState = sys.getCurrentState()

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The initialised Position is not correct!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The initialised temperature is not correct!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The initialised totEnergy is not correct!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The initialised totPotEnergy is not correct!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The initialised totKinEnergy is not correct!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The initialised dhdpos is not correct!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The initialised velocity is not correct!")
        self.assertEqual(curState.lam, expected_state.lam, msg="The initialised lam is not correct!")
        self.assertEqual(curState.dhdlam, expected_state.dhdlam, msg="The initialised dHdlam is not correct!")

    def test_revertStep(self):

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1.0

        newPosition2 = 13
        newVelocity2 = -4
        newForces2 = 8
        newLam2 = 0.5

        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb)
        conditions = []
        temperature = 300
        position = [0]
        mass = [1]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)

        sys.append_state(newPosition=newPosition, newVelocity=newVelocity, newForces=newForces, newLam=newLam)
        expected_state = sys.getCurrentState()
        sys.append_state(newPosition=newPosition2, newVelocity=newVelocity2, newForces=newForces2, newLam=newLam2)
        not_expected_state = sys.getCurrentState()
        sys.revertStep()
        curState = sys.getCurrentState()

        # check current state intialisation
        self.assertEqual(curState.position, expected_state.position, msg="The current Position is not equal to the one two steps before!")
        self.assertEqual(curState.temperature, expected_state.temperature, msg="The current temperature is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totEnergy, expected_state.totEnergy, msg="The current totEnergy is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totPotEnergy, expected_state.totPotEnergy, msg="The current totPotEnergy is not equal to the one two steps before!")
        self.assertAlmostEqual(curState.totKinEnergy, expected_state.totKinEnergy, msg="The current totKinEnergy is not equal to the one two steps before!")
        self.assertEqual(curState.dhdpos, expected_state.dhdpos, msg="The current dhdpos is not equal to the one two steps before!")
        self.assertEqual(curState.velocity, expected_state.velocity, msg="The current velocity is not equal to the one two steps before!")
        self.assertEqual(curState.lam, expected_state.lam, msg="The current lam is not equal to the one two steps before!")
        self.assertEqual(curState.dhdlam, expected_state.dhdlam, msg="The initialised dHdlam is not correct!")

        #check that middle step is not sames
        self.assertNotEqual(curState.position, not_expected_state.position, msg="The not expected Position equals the current one!")
        self.assertEqual(curState.temperature, not_expected_state.temperature, msg="The not expected temperature equals the current one")
        self.assertNotAlmostEqual(curState.totEnergy, not_expected_state.totEnergy, msg="The not expected totEnergy equals the current one")
        self.assertNotAlmostEqual(curState.totPotEnergy, not_expected_state.totPotEnergy, msg="The not expected totPotEnergy equals the current one")
        self.assertNotAlmostEqual(curState.totKinEnergy, not_expected_state.totKinEnergy, msg="The not expected totKinEnergy equals the current one")
        self.assertNotEqual(curState.dhdpos, not_expected_state.dhdpos, msg="The not expected dhdpos, equals the current one")
        self.assertNotEqual(curState.velocity, not_expected_state.velocity, msg="The not expected velocity equals the current one")
        self.assertNotEqual(curState.lam, not_expected_state.lam, msg="The not expected lam equals the current one")
        self.assertEqual(curState.dhdlam, expected_state.dhdlam, msg="The initialised dHdlam is not correct!")

    def test_propergate(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        initialState = sys.getCurrentState()
        sys.propagate()

        #check that middle step is not sames
        self.assertNotEqual(sys._currentPosition, initialState.position, msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertEqual(sys._currentTemperature, initialState.temperature, msg="The initialState does not equal the currentState after propergating in attribute: temperature!")
        self.assertAlmostEqual(sys._currentTotPot, initialState.totPotEnergy, msg="The initialState  does not equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(sys._currentTotKin, initialState.totKinEnergy, msg="The initialState  does not equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertNotEqual(sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity, msg="The initialState does not equal the currentState after propergating in attribute: velocity!")
        self.assertEqual(sys._currentLam, initialState.lam, msg="The initialState does not equal the currentState after propergating in attribute: lam!")
        self.assertEqual(sys._currentdHdLam, initialState.dhdlam, msg="The initialState does not equal the currentState after propergating in attribute: dHdLam!")

    def test_simulate(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)
        steps=100

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        init_state = sys.getCurrentState()
        sys.simulate(steps=steps, initSystem=False, withdrawTraj=True)  #withdrawTraj is needed in the context because of the interaction between different Tests
        trajectory = sys.getTrajectory()

        old_frame = trajectory[0]

        #Check that the first frame is the initial state!
        self.assertEqual(init_state.position, old_frame.position, msg="The initial state does not equal the frame 0 after propergating in attribute: Position!")
        self.assertEqual(init_state.temperature, old_frame.temperature,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: temperature!")
        self.assertAlmostEqual(init_state.totPotEnergy, old_frame.totPotEnergy,
                                  msg="The initial state does not equal the frame 0 after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(init_state.totKinEnergy, old_frame.totKinEnergy,
                                  msg="The initial state does not equal the frame 0 after propergating in attribute: totKinEnergy!")
        self.assertEqual(init_state.dhdpos, old_frame.dhdpos, msg="The initial state does not equal the frame 0 after propergating in attribute: dhdpos!")
        self.assertEqual(init_state.velocity, old_frame.velocity,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: velocity!")
        self.assertEqual(init_state.lam, old_frame.lam,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: lam!")
        self.assertEqual(init_state.dhdlam, old_frame.dhdlam,
                            msg="The initial state does not equal the frame 0 after propergating in attribute: dhdLam!")

        #check that the frames are all different from each other.
        for ind, frame in enumerate(trajectory[1:]):
            #check that middle step is not sames
            self.assertNotEqual(old_frame.position, frame.position, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: Position!")
            self.assertEqual(old_frame.temperature, frame.temperature, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: temperature!")    #due to integrator
            self.assertNotAlmostEqual(old_frame.totPotEnergy, frame.totPotEnergy, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: totPotEnergy!")
            self.assertAlmostEqual(old_frame.totKinEnergy, frame.totKinEnergy, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: totKinEnergy!")    #due to integrator
            self.assertNotEqual(old_frame.dhdpos, frame.dhdpos, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: dhdpos!")
            self.assertEqual(old_frame.velocity, frame.velocity, msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: velocity!")     #due to integrator
            self.assertEqual(init_state.lam, old_frame.lam,
                             msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: lam!")
            self.assertEqual(init_state.dhdlam, old_frame.dhdlam,
                             msg="The frame "+str(ind)+" equals the frame  "+str(ind+1)+" after propergating in attribute: dhdLam!")
            old_frame = frame

    def test_applyConditions(self):
        """
        NOT IMPLEMENTED!
        """
        pass

    def test_initVel(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        sys.initVel()

        cur_velocity = sys._currentVelocities

        self.assertEqual(type(cur_velocity), type(np.array([1])),msg="Velocity has not the correcttype!")
        self.assertTrue(all(map(lambda vel: isinstance(vel, numbers.Number),cur_velocity)),msg="Velocity has not the correcttype!")

    def test_updateTemp(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def test_updateEne(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        initialState = sys.getCurrentState()
        sys.propagate()
        sys.updateEne()

        #check that middle step is not sames
        self.assertNotEqual(sys._currentPosition, initialState.position, msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertEqual(sys._currentTemperature, initialState.temperature, msg="The initialState does not equal the currentState after propergating in attribute: temperature!")
        self.assertNotAlmostEqual(sys._currentTotPot, initialState.totPotEnergy, msg="The initialState  does equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertAlmostEqual(sys._currentTotKin, initialState.totKinEnergy, msg="The initialState  does equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertNotEqual(sys._currentForce, initialState.dhdpos, msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity, msg="The initialState does not equal the currentState after propergating in attribute: velocity!")

    def test_totPot(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb, lam=lam)

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        self.assertAlmostEqual(sys.totPot(), 12.5, msg="The initialised totPotEnergy is not correct!")

    def test_totKin(self):
        """
        uses init_state, updateEne, randomPos, self.state
        :return:
        """
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb)

        temperature = 300
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        self.assertAlmostEqual(sys.totKin(), 0.0, msg="The initialised totPotEnergy is not correct!")

        newPosition = 10
        newVelocity = -5
        newForces = 3
        newLam = 1
        sys.append_state(newPosition= newPosition,newVelocity= newVelocity, newForces=newForces, newLam=newLam)
        self.assertAlmostEqual(sys.totKin(), 12.5, msg="The initialised totPotEnergy is not correct!")

    def test_setTemperature(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb)

        temperature = 300
        temperature2 = 600
        position = [0]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        sys._currentVelocities = 100
        sys.updateCurrentState()
        initialState = sys.getCurrentState()
        sys.set_Temperature(temperature2)

        # check that middle step is not sames
        self.assertEqual(sys._currentPosition, initialState.position,
                            msg="The initialState equals the currentState after propergating in attribute: Position!")
        self.assertNotEqual(sys._currentTemperature, initialState.temperature,
                         msg="The initialState does equal the currentState after propergating in attribute: temperature!")
        self.assertAlmostEqual(sys._currentTotPot, initialState.totPotEnergy,
                                  msg="The initialState  does equal  the currentState after propergating in attribute: totPotEnergy!")
        self.assertNotAlmostEqual(sys._currentTotKin, initialState.totKinEnergy,
                               msg="The initialState  does not equal  the currentState after propergating in attribute: totKinEnergy!")
        self.assertEqual(sys._currentForce, initialState.dhdpos,
                            msg="The initialState equals the currentState after propergating in attribute: dhdpos!")
        self.assertEqual(sys._currentVelocities, initialState.velocity,
                         msg="The initialState does equal the currentState after propergating in attribute: velocity!")

    def test_get_Pot(self):
        integ = integrator.monteCarloIntegrator()
        ha = potential1D.harmonicOsc(x_shift=-5)
        hb = potential1D.harmonicOsc(x_shift=5)
        lam = 0
        pot = potentials.OneD.linCoupledHosc(ha=ha, hb=hb)

        temperature = 300
        position = [5]

        sys = system.perturbedSystem(potential=pot, integrator=integ, position=position, temperature=temperature, lam=lam)
        self.assertEqual(50.0, sys.getPot()[0], msg="Could not get the correct Pot Energy!")


"""
class testSystem(unittest.TestCase):
    def testFc(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0
        for fc in [1.0, 1.5, 10.0, 20.0]:
            with self.subTest(fc=fc):
                sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                    potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha))
                sys.pos = 1.0
                sys.updateEne()
                self.assertAlmostEqual(sys.totpot, fc)
                self.assertAlmostEqual(sys.totPot.dhdpos(sys.lam, sys.pos), 2.0 * fc)
                self.assertAlmostEqual(sys.totPot.dhdl(sys.lam, sys.pos), 0.5 * fc)

    def testAlpha(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0

        for alpha in [1.0, 10.0, 100.0]:
            with self.subTest(alpha=alpha):
                potential = pot.pertHarmonicOsc1D(fc=fc, alpha=alpha)
                sys = system.perturbedSystem(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass, potential=potential)
                sys.pos = 1.0
                sys.updateEne()
                self.assertAlmostEqual(sys.totpot, 0.5 + 0.5 * alpha)
                self.assertAlmostEqual(sys.totPot.dhdpos(sys.lam, sys.pos), 1.0 + alpha)
                self.assertAlmostEqual(sys.totPot.dhdl(sys.lam, sys.pos), 0.5 * alpha)

    def testLam(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0

        for lam in [0.0, 0.5, 1.0]:
            with self.subTest(lam=lam):
                sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                    potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha))
                sys.pos = 1.0
                sys.updateEne()
                self.assertAlmostEqual(sys.totpot, 0.5 + 0.5 * lam)
                self.assertAlmostEqual(sys.totPot.dhdpos(sys.lam, sys.pos), 1.0 + lam)
                self.assertAlmostEqual(sys.totPot.dhdl(sys.lam, sys.pos), 0.5)

    def testKin(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0

        for temp in [250, 300, 350]:
            with self.subTest(temp=temp):
                for sd in [0, 1]:
                    for nose in [0, 1]:
                        for mass in [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]]:
                            sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                                potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha))

    def testNoseHoover(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0
        for vel in [1.0, 2.0, 3.0]:
            with self.subTest(vel=vel):
                sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                    potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha))
                sys.vel = vel
                sys.updateEne()
                self.assertAlmostEqual(sys.totkin, vel ** 2 / 4.0)
                sys.propagate()

    def testNewton(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0
        for vel in [1.0, 2.0, 3.0]:
            with self.subTest(vel=vel):
                sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                    potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha), integrator='newton')
                sys.pos = 1.0
                sys.newpos = 1.0
                sys.newvel = vel
                sys.vel = vel
                sys.updateEne()
                self.assertAlmostEqual(sys.totkin, vel ** 2 / 4.0)
                sys.propagate()
                self.assertAlmostEqual(sys.newvel, vel - 0.4)
                self.assertAlmostEqual(sys.vel, vel - 0.2)
                self.assertAlmostEqual(sys.newpos, 1.0 + (vel - 0.4) * 0.1)
        vel = 1.0
        sys = sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                                  potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha), integrator='newton')
        sys.pos = 1.0
        sys.newpos = 1.0
        sys.vel = 1.0
        sys.newvel = 1.0
        for i, j in zip(
                [1.0, 0.8, 3.848716665874072e-05, 0.16000000000000003, 1.0, 1.1600000000000001, 1.0, 0.0],
                sys.propagate()):
            self.assertAlmostEqual(i, j)

    def testHMC(self):
        temp = 300.0
        mass = [1.0, 1.0]
        alpha = 1.0
        lam = 1.0
        fc = 1.0
        sys = system.system(temp=temp, fc=fc, lam=lam, alpha=alpha, mass=mass,
                            potential=pot.pertHarmonicOsc1D(fc=fc, alpha=alpha), integrator='hmc')

"""
if __name__ == '__main__':
    unittest.main()

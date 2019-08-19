"""
Module: System
    This module shall be used to implement subclasses of system1D. It wraps all information needed and generated by a simulation.
"""

import numpy as np
from typing import Iterable
from numbers import Number
import pandas as pd
import scipy.constants as const

from Ensembler.src import dataStructure as data
from Ensembler.src.potentials.OneD import _potential1DCls as _potentialCls, _perturbedPotential1DCls as _perturbedPotentialCls

from Ensembler.src.integrator import _integratorCls
from Ensembler.src.conditions.conditions import Condition

class system:
    '''
    ..autoclass:: Class of a system on a
    potential energy surface defined by a potential1D._potentialCls object or other.
    Different integrators/propagators can be chosen to create an ensemble of states.
    Also other parameters are tunable.
    '''
    state = data.basicState

    #settings:
    potential:_potentialCls = None
    integrator:_integratorCls = None
    conditions=[]
    
    #parameters
    temperature:float = 298.0
    mass:float = 1 #for one particle systems!!!!
    nparticles:int =1 #Todo: adapt it to be multiple particles
    nDim:int=-1
    initial_positions:Iterable[float]
    
    #output
    currentState:state = None
    trajectory:list = []
        
    #tmpvars - private:
    _currentTotPot:(Number or Iterable[Number]) = None
    _currentTotKin:(Number or Iterable[Number]) = None
    _currentPosition:(Number or Iterable[Number]) = None
    _currentVelocities:(Number or Iterable[Number]) = None
    _currentForce:(Number or Iterable[Number]) = None
    _currentTemperature:(Number or Iterable[Number]) = None

    def __init__(self, potential:_potentialCls, integrator:_integratorCls, conditions:Iterable[Condition]=[],
                 temperature:float=298.0, position:(Iterable[float] or float)=None, mass:float=1):

        #params
        self.potential = potential
        self.integrator = integrator
        self.conditions = conditions
        self.temperature = temperature
        self.mass = mass
        self.nDim = potential.nDim

        #is the potential a state dependent one? - needed for initial pos.
        if(hasattr(potential, "nStates")):
            self.nStates = potential.nStates
            if(hasattr(potential, "states_coupled")):   #does each state get the same position?
                self.states_coupled = potential.states_coupled
            else:
                self.states_coupled = True #Todo: is this a good Idea?
        else:
            self.nstates = 1

        #Settings for potential
        potential._set_no_type_check()  #initially taken care by system. Saves performance!
        potential.set_singlePos_mode()  #easier execution... does apparently not save so much performacne

        #check if system is coupled to conditions:
        for condition in self.conditions:
            if(not hasattr(condition, "system")):
                condition.coupleSystem(self)
            if(not hasattr(condition, "dt") and hasattr(self.integrator, "dt")):
                condition.dt = self.integrator.dt
            else:
                condition.dt=1

        #define dims of system. #Todo: only true for one dim Pot.
        self.init_state(initial_position=position)

    def init_state(self, initial_position=None):
        #initial position given?
        if (initial_position == None):
            self.initial_positions  = self.randomPos()
        else:
            self.initial_positions = initial_position

        self._currentForce = self.potential.dhdpos(self.initial_positions)  #initialise forces!

        #set a new current_state
        self.set_current_state(currentPosition=self.initial_positions, currentVelocities=self._currentVelocities, currentForce=self._currentForce, currentTemperature=self.temperature)

    def randomPos(self)-> Iterable:
        if(self.nStates==1):
            return np.subtract(np.multiply(np.random.rand(self.nDim),20),10)
        elif(self.nStates >1 and self.states_coupled):
            position = np.subtract(np.multiply(np.random.rand(self.nDim), 20), 10)
            return [position for state in range(self.nStates)]
        else:
            return [np.subtract(np.multiply(np.random.rand(self.nDim), 20), 10) for state in range(self.nStates)]

    def append_state(self, newPosition, newVelocity, newForces):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces

        self.updateTemp()
        self.updateEne()
        self.updateCurrentState()

        self.trajectory.append(self.currentState)

    def propagate(self):
        self._currentPosition, self._currentVelocities, self._currentForce = self.integrator.step(self) #self.current_state)

    def revertStep(self):
        self.currentState = self.trajectory[-2]
        return

    def simulate(self, steps:int, initSystem:bool=True, withdrawTraj:bool=False, save_every_state:int=1):
        if(steps > 1000):
            show_progress =True
            block_length = steps*0.1
        else:
            show_progress = False

        if(withdrawTraj):
            self.trajectory = []
            
        if(initSystem): #type(self._currentVelocities) == type(None) or type(self._currentPosition) == type(None)
            self.initVel()
            self.init_state(initial_position=self.initial_positions)

        self.updateCurrentState()

        if(show_progress): print("Progress: ", end="\t")
        step = 0
        for step in range(steps):
            if(show_progress and step%block_length==0):
                print(str(100*step//steps)+"%", end="\t")

            if(step%save_every_state == 0 ):
                self.trajectory.append(self.currentState)

            #Do one simulation Step. Todo: change to do multi steps
            self.propagate()

            #Calc new Energy
            self.updateEne()

            #Apply Restraints, Constraints ...
            self.applyConditions()

            #Set new State
            self.updateCurrentState()

        if(step%save_every_state != 0 ):
            self.trajectory.append(self.currentState)

        if(show_progress): print("100%")
        return self.currentState

    def applyConditions(self):
        for aditional in self.conditions:
            aditional.apply()

    def initVel(self):
        self._currentVelocities = np.array([np.sqrt(const.gas_constant / 1000.0 * self.temperature / self.mass) * np.random.normal() for dim in range(self.nDim)])
        self.veltemp = self.mass / const.gas_constant / 1000.0 * np.linalg.norm(self._currentVelocities) ** 2  # t
        return self._currentVelocities

    def updateTemp(self):
        """ this looks like a thermostat like thing! not implemented!@ TODO calc velocity from speed"""
        self._currentTemperature = self._currentTemperature

    def updateEne(self):
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()

    def updateCurrentState(self):
        self.currentState = self.state(self._currentPosition, self._currentTemperature,
                                        (self._currentTotKin+self._currentTotPot),
                                        self._currentTotPot, self._currentTotKin,
                                        self._currentForce, self._currentVelocities)

    def totKin(self):
        if(self._currentVelocities != None):
            return 0.5 * self.mass * np.square(np.linalg.norm(self._currentVelocities))  #Todo: velocities is a VECTOR not sum but vector length is needed
        else:
            return 0

    def totPot(self)->float:
        return float(np.sum(self.potential.ene(self._currentPosition)))

    def getPot(self)->Iterable[float]:
        return self.potential.ene(self._currentPosition)

    def getTotEnergy(self):
        self.updateEne()
        return np.add(self.totPot(), self.totKin())

    def getCurrentState(self)->state:
        return self.currentState
    
    def getTrajectoryObjects(self)->Iterable[state]:
        return self.trajectory

    def getTrajectory(self)->pd.DataFrame:
        return pd.DataFrame.from_dict([frame._asdict() for frame  in self.trajectory])

    def set_current_state(self, currentPosition:(Number or Iterable), currentVelocities:(Number or Iterable)=0, currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature
        self.currentState = self.state(self._currentPosition, self._currentTemperature,0, 0, 0, 0, 0)

        self.updateEne()
        self.updateCurrentState()

    def set_Temperature(self, temperature):
        """ this looks like a thermostat like thing! not implemented!@"""
        self.temperature = temperature
        self._currentTemperature = temperature
        self.updateEne()

class perturbedSystem(system):
    """
    
    """

    #Lambda Dependend Settings
    state = data.lambdaState
    currentState: data.lambdaState
    potential: _perturbedPotentialCls
    #current lambda
    _currentLam:float = None
    _currentdHdLam:float = None

    def __init__(self, potential:_perturbedPotentialCls, integrator: _integratorCls, conditions: Iterable[Condition]=[],
                 temperature: float = 298.0, position:(Iterable[Number] or float) = None, lam:float=0.0):

        self._currentLam = lam
        if(not isinstance(potential, _perturbedPotentialCls)):
            raise Exception("Potential: "+potential.name+" is not a potential of the _perturbedPotentialCls family! Please use these types for: "+__class__.__name__)
        else:
            if(hasattr(potential, "lam")):
                setattr(potential, "lam", lam)
            else:
                Exception(
                    "Potential: " + potential.name + " has not an attribute lam. But this attribute is needed for representing lambda! Please add the field for: " + __class__.__name__)

        super().__init__(potential=potential, integrator=integrator, conditions=conditions,
                 temperature=temperature, position=position)


    def init_state(self, initial_position=None):
        #initial position given?
        if (initial_position == None):
            self.initial_positions  = self.randomPos()
        else:
            self.initial_positions = initial_position

        #set a new current_state
        self.set_current_state(currentPosition=self.initial_positions, currentVelocities=self._currentVelocities, currentForce=self._currentForce, currentTemperature=self.temperature, currentLambda=self._currentLam, currentdHdLam=self._currentdHdLam)

    def set_current_state(self, currentPosition:(Number or Iterable), currentLambda:(Number or Iterable), currentVelocities:(Number or Iterable)=0,  currentdHdLam:(Number or Iterable)=0,
                          currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=0,
                                       totPotEnergy=0, totKinEnergy=0,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       lam=self._currentLam, dhdlam=0)

        self.updateEne()
        self.updateCurrentState()

    def updateCurrentState(self):
        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=(self._currentTotKin + self._currentTotPot),
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       lam=self._currentLam, dhdlam=self._currentdHdLam)



    """
    def updateEne(self):
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()
        self._currentTotE = self._currentTotKin + self._currentTotPot
        self.redene = np.divide(self._currentTotE, np.divide(const.gas_constant, np.multiply(1000.0, self._currentPosition)))
    """
    def append_state(self, newPosition, newVelocity, newForces, newLam):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces
        self._currentLam = newLam

        self.updateTemp()
        self.updateEne()
        self.updateCurrentState()
        self.trajectory.append(self.currentState)

    def set_lambda(self, lam):
        self._currentLam = lam
        #self.omega = np.sqrt((1.0 + self.potential.alpha * self._currentLam) * self.potential.fc / self.mass)
        self.potential.set_lam(lam=self._currentLam)
        self.updateEne()
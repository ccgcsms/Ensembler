"""
Module: System
    This module shall be used to implement subclasses of system1D. It wraps all information needed and generated by a simulation.
"""

import numpy as np
from typing import Iterable
from numbers import Number
import scipy.constants as const
from Ensembler.src import dataStructure as data
from Ensembler.src.potentials.OneD import _potential1DCls as _potentialCls, _perturbedPotential1DCls as _perturbedPotentialCls

from Ensembler.src.integrator import integrator as integratorCls
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
    integrator:integratorCls = None
    conditions=[]
    
    #parameters
    temperature:float = 298.0
    mass:float = 1 #for one particle systems!!!!
    nparticles:int =1 #Todo: adapt it to be multiple particles
    nDim:int
    initial_positions:Iterable[float]
    
    #output
    currentState:state = None
    trajectory:Iterable = []
        
    #tmpvars - private:
    _currentTotPot:float = None
    _currentTotKin:float = None
    _currentPosition:float = None
    _currentVelocities:float = None
    _currentForce:float = None
    _currentTemperature:float = None
        
    def __init__(self, potential:_potentialCls, integrator:integratorCls, conditions:Iterable[Condition]=[],
                 temperature:float=298.0, position:float=None, mass:float=1):

        #params
        self.potential = potential
        self.integrator = integrator
        self.conditions = conditions
        self.temperature = temperature
        self.mass = mass

        self.nDim = potential.nDim



        #check if system is coupled to conditions:
        for condition in self.conditions:
            if(not hasattr(condition, "system")):
                condition.coupleSystem(self)
            if(not hasattr(condition, "dt") and hasattr(self.integrator, "dt")):
                condition.dt = self.integrator.dt
            else:
                condition.dt=1
        #define dims of system. #Todo: only true for one dim Pot.
        self.nDim = potential.nDim
        self.init_state(initial_position=position)

    def init_state(self, initial_position=None):
        if (initial_position == None):
            initial_position = self.randomPos()

        self.initial_positions = initial_position

        self._currentPosition = self.initial_positions
        self._currentForce = 0
        self._currentVelocities = [0 for dim in range(self.potential.nDim)]
        self._currentTemperature = self.temperature
        self.currentState = self.state(self._currentPosition, self._currentTemperature,0, 0, 0, 0, 0)

        self.updateEne()
        
        self.currentState =  self.state(self._currentPosition, self.temperature,
                                        (self._currentTotKin+self._currentTotPot),
                                        self._currentTotPot, self._currentTotKin,
                                        self._currentForce, self._currentVelocities)

    def randomPos(self)-> Iterable[Iterable[Number]]:
        pos = [[np.random.rand() * 20.0 - 10.0 for x in range(self.nDim)]]
        return pos

    def append_state(self, newPosition, newVelocity, newForces):
        self._currentPosition = newPosition
        self._currentVelocities = newVelocity
        self._currentForce = newForces

        self.updateEne()
        self.updateTemp()

        self.currentState = self.state(self._currentPosition, self.temperature,
                                       self._currentTotKin + self._currentTotPot,
                                       self._currentTotPot, self._currentTotKin,
                                       self._currentForce, self._currentVelocities)

        self.trajectory.append(self.currentState)

    def simulate(self, steps:int, initSystem:bool=True, withdrawTraj:bool=False, save_every_state:int=1):
        
        if(steps > 1000):
            show_progress =True
            block_length = steps*0.1
        else:
            show_progress = False

        if(withdrawTraj):
            self.trajectory = []
            
        if(type(self._currentVelocities) == type(None) or initSystem):
            self.initVel()
        
        if(type(self._currentPosition) == type(None) or initSystem):
            self._currentPosition = self.randomPos()
        
        self.updateEne()    #inti E
        if(show_progress): print("Progress: ", end="\t")
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

            self.currentState =  self.state(self._currentPosition, self.temperature,
                                            self._currentTotKin + self._currentTotPot,
                                            self._currentTotPot, self._currentTotKin,
                                            self._currentForce, self._currentVelocities)


        if(show_progress): print("100%")
        return self.currentState

    def applyConditions(self):
        for aditional in self.conditions:
            aditional.apply()


    def randomShift(self):
        posShift = (np.random.rand() * 2.0 - 1.0) * self.posShift
        return posShift

    def initVel(self):
        self._currentVelocities = np.array([np.sqrt(const.gas_constant / 1000.0 * self.temperature / self.mass) * np.random.normal() for dim in range(self.nDim)])
        self.veltemp = self.mass / const.gas_constant / 1000.0 * np.linalg.norm(self._currentVelocities) ** 2  # t
        return self._currentVelocities

    def updateTemp(self, temperature:float):
        self.temperature = temperature
        self.alpha = 1.0
        self.posShift = np.sqrt(0.5 * (1.0 + self.alpha * 0.5) * 1 / (const.gas_constant / 1000.0 * self.temperature))
        self.updateEne()

    def updateEne(self):
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()

    def totKin(self):
        if(self._currentVelocities != None):
            return 0.5 * self.mass * np.linalg.norm(self._currentVelocities) ** 2  #Todo: velocities is a VECTOR not sum but vector length is needed
        else:
            return 0

    def totPot(self)->float:
        return sum(map(sum, self.potential.ene(self._currentPosition)))

    def getPot(self)->Iterable[float]:
        return self.potential.ene(self._currentPosition)

    def getTotEnergy(self):
        self.updateEne()
        return self.totPot()+self.totKin()

    def propagate(self):
        self._currentPosition, self._currentVelocities, self._currentForce = self.integrator.step(self) #self.current_state)

    def revertStep(self):
        self.currentState = self.trajectory[-2]
        return
    
    def getCurrentState(self)->state:
        return self.currentState
    
    def getTrajectory(self)->Iterable[state]:
        return self.trajectory

class perturbedSystem(system):
    """
    
    """

    #
    state = data.lambdaState
    currentState: data.lambdaState
    potential: _perturbedPotentialCls

    #
    _currentLam:float = None

    def __init__(self, potential:_potentialCls, integrator: integratorCls, conditions: Iterable[Condition]=[],
                 temperature: float = 298.0, position: float = None, lam:float=0.0):

        self._currentLam = lam
        super().__init__(potential=potential, integrator=integrator, conditions=conditions,
                 temperature=temperature, position=position)


    def init_state(self):
        self.currentState = self.state(position=self._currentPosition, temperature=0,
                                       totEnergy=0, totPotEnergy=0, totKinEnergy=0,
                                       dhdpos=0, velocity=0,
                                       lamb=self._currentLam, dhdlam=0)
        self.updateEne()

        self.currentState = self.state(position=self._currentPosition, temperature=self.temperature,
                                       totEnergy=(self._currentTotKin + self._currentTotPot),
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       lamb=self._currentLam, dhdlam=0)

        
    def updateEne(self):
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()
        self._currentTotE = self._currentTotKin + self._currentTotPot
        self.redene = self._currentTotE / (const.gas_constant / 1000.0 * self._currentPosition)

    def updateLam(self, lam):
        self._currentLam = lam
        self.omega = np.sqrt((1.0 + self.potential.alpha * self._currentLam) * self.potential.fc / self.mass)
        self.potential.set_lam(lam=self._currentLam)
        self.updateEne()

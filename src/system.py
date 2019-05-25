"""
Module: System
    This module shall be used to implement subclasses of system1D. It wraps all information needed and generated by a simulation.
"""

import numpy as np
from typing import List
import scipy.constants as const
import src.dataStructure as data
from src.potential1D import potentialCls
from src.integrator import integrator as integratorCls
from src.conditions import condition

class system:
    '''
    Class of a (possibly perturbed) 1D system on a
    potential energy surface defined by a potential1D.potential object.
    Different integrators/propagators can be chosen to create an ensemble of states.
    '''
    state = data.basicState

    #settings:
    potential:potentialCls = None
    integrator:integratorCls = None
    conditions=[]
    
    #parameters
    temperature:float = 298.0
    mass:float = 1 #for one particle systems!!!!
    initial_positions:List[float]
    
    #output
    currentState:state = None
    trajectory:List = []
        
    #tmpvars - private:
    _currentTotPot:float = None
    _currentTotKin:float = None
    _currentPosition:float = None
    _currentVellocities:float = None
    _currentForce:float = None
    _currentTemperature:float = None
        
    def __init__(self, potential:potentialCls, integrator:integratorCls, conditions:List[condition]=[],
                 temperature:float=298.0, position:float=None, mass:float=1):
        self.potential = potential
        self.integrator = integrator
        self.conditions = conditions
        self.temperature = temperature
        self.mass = mass

        #params
        self.initial_positions = position
        
        #dummy for calc E
        if(position == None):
            position = self.randomPos()

        self._currentPosition = position
        self._currentTemperature = temperature

        self.init_state()

    def init_state(self):
        self._currentForce = 0
        self._currentVellocities = 0
        self.currentState = self.state(self._currentPosition, self._currentTemperature,0, 0, 0, 0, 0)

        self.updateEne()
        
        self.currentState =  self.state(self._currentPosition, self.temperature,
                                        (self._currentTotKin+self._currentTotPot),
                                        self._currentTotPot, self._currentTotKin,
                                        self._currentForce, self._currentVellocities)

    def append_state(self, newPosition, newVelocity, newForces):
        self._currentPosition = newPosition
        self._currentVellocities = newVelocity
        self._currentForce = newForces

        self.updateEne()
        self.updateTemp()

        self.currentState = self.state(self._currentPosition, self.temperature,
                                       self._currentTotKin + self._currentTotPot,
                                       self._currentTotPot, self._currentTotKin,
                                       self._currentForce, self._currentVellocities)

        self.trajectory.append(self.currentState)

    def simulate(self, steps:int, initSystem:bool=True, withdrawTraj:bool=False, save_every_state:int=1):
        
        if(steps > 1000):
            show_progress =True
            block_length = steps*0.1
        else:
            show_progress = False

        if(withdrawTraj):
            self.trajectory = []
            
        if(self._currentVellocities == None or initSystem):
            self.initVel()
        
        if(self._currentPosition == None or initSystem):
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
                                            self._currentForce, self._currentVellocities)


        if(show_progress): print("100%")
        return self.currentState

    def applyConditions(self):
        for aditional in self.conditions:
            aditional.do(self.current_state)

    def randomPos(self):
        pos = np.random.rand() * 20.0 - 10.0
        return pos

    def randomShift(self):
        posShift = (np.random.rand() * 2.0 - 1.0) * self.posShift
        return posShift

    def initVel(self):
        self._currentVellocities = np.sqrt(const.gas_constant / 1000.0 * self.temperature / self.mass) * np.random.normal()
        self.veltemp = self.mass / const.gas_constant / 1000.0 * self._currentVellocities ** 2  # t
        return self._currentVellocities

    def updateTemp(self, temperature:float):
        self.temperature = temperature
        self.alpha = 1.0
        self.posShift = np.sqrt(0.5 * (1.0 + self.alpha * 0.5) * 1 / (const.gas_constant / 1000.0 * self.temperature))
        self.updateEne()

    def updateEne(self):
        self._currentTotPot = self.totPot()
        self._currentTotKin = self.totKin()
       # self.redene = self.totpot / (const.gas_constant / 1000.0 * self.temp)#?
    
    def totKin(self):
        if(self._currentVellocities != None):
            return 0.5 * self.mass * self._currentVellocities ** 2
        else:
            return 0

    def totPot(self)->float:
        return sum(self.potential.ene(self._currentPosition))

    def getPot(self)->List[float]:
        return self.potential.ene(self._currentPosition)

    def propagate(self):
        self._currentPosition, self._currentVellocities, self._currentForce = self.integrator.step(self) #self.current_state)

    def revertStep(self):
        self.current_state = self.trajectory[-2]
        return
    
    def getCurrentState(self)->state:
        return self.currentState
    
    def getTrajectory(self)->List[state]:
        return self.trajectory

    
    
class perturbedtSystem(system):
    """
    
    """
    state = data.lambdaState

    def __init__(self, potential:potentialCls, integrator: integratorCls, conditions: List[condition],
                 temperature: float = 298.0, position: float = None, lamb:float=0.0):

        self.lamb = lamb
        super().__init__(potential=potential, integrator=integrator, conditions=conditions,
                 temperature=temperature, position=position)


    def init_state(self):
        self.currentState = self.state(position=self._currentPosition, temperature=0,
                                       totEnergy=0, totPotEnergy=0, totKinEnergy=0,
                                       dhdpos=0, velocity=0,
                                       lamb=self.lamb, dhdlam=0)
        self.updateEne()

        self.currentState = self.state(position=self._currentPosition, temperature=self.temperature,
                                       totEnergy=(self._currentTotKin + self._currentTotPot),
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVellocities,
                                       lamb=self.lamb, dhdlam=0)

        
    def updateEne(self):
        self.current_state = self.state(position=self.pos, temperature=self.temp,
                                        totEnergy=self.totene, totPotEnergy=self.totPot.ene(self.lam, self.pos), totKinEnergy=self.totkin)

        self.totpot = self.totPot.ene(self.lam, self.pos)
        self.dhdlam = self.totPot.dhdlam(self.lam, self.pos)
        self.totkin = self.totKin()
        self.totene = self.totpot + self.totkin
        self.redene = self.totpot / (const.gas_constant / 1000.0 * self.temp)
        
        
    def update(self, lam):
        self.lam = lam
        self.omega = np.sqrt((1.0 + self.alpha * self.lam) * self.fc / self.mass)
        self.updateEne()
        
def edsSystem(system1D):
    pass

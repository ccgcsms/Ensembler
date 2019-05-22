"""
Module: System
    This module shall be used to implement subclasses of system1D. It wraps all information needed and generated by a simulation.
"""

import numpy as np
from typing import List
import scipy.constants as const
import src.dataStructure as data
from src.potential1D import potential
from src.integrator import integrator as integratorCls
from src.conditions import condition

class system1D:
    '''
    Class of a (possibly perturbed) 1D system on a
    potential energy surface defined by a potential1D.potential object.
    Different integrators/propagators can be chosen to create an ensemble of states.
    '''
    state = data.basicState

    #settings:
    potential:potential = None
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
    _currentVel:float = None
    _currentForce:float = None
    _currentTemperature:float = None
        
    def __init__(self, potential:potential, integrator:integratorCls, conditions:List[condition],
                 temperature:float=298.0, position:float=None):
        self.potential = potential
        self.integrator = integrator
        self.temperature = temperature
        
        #params
        self.initial_positions = position
        
        #dummy for calc E
        if(position == None):
            position = self.randomPos()
            
        self.currentState = self.state(position, temperature,0, 0, 0, 0, 0)
        self._currentPosition = position
        self._currentTemperature = temperature
               
        self.updateEne()
        
        print(self._currentTotPot, self._currentTotKin)
        self.currentState =  self.state(self._currentPosition, self.temperature,
                                       (self._currentTotKin+self._currentTotPot), 
                                        self._currentTotPot, self._currentTotKin,
                                        self._currentForce, self._currentVel)
       

        
    def simulate(self, steps:int, initSystem:bool=False, withdrawTraj:bool=False):
        
        if(steps > 1000):
            show_progress =True
            block_length = steps*0.1
        else:
            show_progress = False
        if(withdrawTraj):
            self.trajectory = []
            
        if(self._currentVel == None or initSystem):
            self.initVel()
        
        if(self._currentPosition == None or initSystem):
            self._currentPosition = self.randomPos()
        
        self.updateEne()    #inti E
        if(show_progress): print("Progress: ", end="\t")
        for step in range(steps):
            if(show_progress and step%block_length==0):
                print(str(100*step//steps)+"%", end="\t")
                
            self.propagate()
            self.updateEne()
            for aditional in self.conditions:
                aditional.do(self.current_state)
            
            self.currentState =  self.state(self._currentPosition, self.temperature,
                                       self._currentTotKin+self._currentTotPot, 
                                       self._currentTotPot, self._currentTotKin,
                                       self._currentForce, self._currentVel)
            self.trajectory.append(self.currentState)
        if(show_progress): print("100%")
        return self.currentState
    
    def randomPos(self):
        pos = np.random.rand() * 20.0 - 10.0
        return pos

    def randomShift(self):
        posShift = (np.random.rand() * 2.0 - 1.0) * self.posShift
        return posShift

    def initVel(self):
        self._currentVel = np.sqrt(const.gas_constant / 1000.0 * self.temperature / self.mass) * np.random.normal()
        self.veltemp = self.mass / const.gas_constant / 1000.0 * self._currentVel ** 2  # t
        return self._currentVel

    def updateTemp(self, temperature:float):
        self.temperature = temperature
        self.alpha = 1.0
        self.posShift = np.sqrt(0.5 * (1.0 + self.alpha * 0.5) * 1 / (const.gas_constant / 1000.0 * self.temperature))
        self.updateEne()

    def updateEne(self):
        self._currentTotPot = self.pot()
        self._currentTotKin = self.kin()
       # self.redene = self.totpot / (const.gas_constant / 1000.0 * self.temp)#?
    
    def kin(self):
        if(self._currentVel != None):
            return 0.5 * self.mass * self._currentVel ** 2
        else:
            return 0

    def pot(self)->float:
        return sum(self.potential.ene(self._currentPosition))
        
    def propagate(self):
        self._currentPosition, self._currentVel, self._currentForce = self.integrator.step(self) #self.current_state)

    def revertStep(self):
        self.current_state = self.trajectory[-2]
        return
    
    def getCurrentState(self)->state:
        return self.currentState
    
    def getTrajectory(self)->List[state]:
        return self.trajectory

    
    
class perturbedtSystem(system1D):
    """
    
    """
    state = data.lambdaState
    def __init__(self,potential:potential, integrator='sd', temp:float=298.0, 
                 fc=1.0, lam=0.0, alpha=10.0, mass=None):
            pass
        
        
    def updateEne(self):
        self.current_state = self.state(position=self.pos, temperature=self.temp,
                     totEnergy=self.totene, totPotEnergy=self.pot.ene(self.lam, self.pos), totKinEnergy=self.totkin)

        self.totpot = self.pot.ene(self.lam, self.pos)
        self.dhdlam = self.pot.dhdlam(self.lam, self.pos)
        self.totkin = self.kin()
        self.totene = self.totpot + self.totkin
        self.redene = self.totpot / (const.gas_constant / 1000.0 * self.temp)
        
        
    def update(self, lam):
        self.lam = lam
        self.omega = np.sqrt((1.0 + self.alpha * self.lam) * self.fc / self.mass)
        self.updateEne()
        
def edsSystem(system1D):
    pass
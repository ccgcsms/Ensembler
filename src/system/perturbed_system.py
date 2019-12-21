
import numpy as np
from typing import Iterable, NoReturn
from numbers import Number
import pandas as pd
import scipy.constants as const
pd.options.mode.use_inf_as_na = True


from src import dataStructure as data
from src.potentials.ND import envelopedPotential
from src.potentials._baseclasses import _potentialNDCls as _potentialCls
from src.potentials._baseclasses import _perturbedPotentialNDCls as _perturbedPotentialCls

from src.integrator import _integratorCls
from src.conditions._conditions import Condition

class perturbedSystem(system):
    """
    
    """

    #Lambda Dependend Settings
    state = data.lambdaState
    currentState: data.lambdaState
    potential: _perturbedPotentialCls
    #current lambda
    _currentLam:float = np.nan
    _currentdHdLam:float = np.nan


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

        super().__init__(potential=potential, integrator=integrator, conditions=conditions, temperature=temperature, position=position)


    def init_position(self, initial_position=None):
        #initial position given?
        if (type(initial_position) == type(None) or np.isnan(initial_position)):
            self.initial_positions = self.randomPos()
        else:
            self.initial_positions = self.potential._check_positions_type_singlePos(initial_position)

        #self._currentForce = self.potential.dhdpos(self.initial_positions)  #initialise forces!    #todo!

        #set a new current_state
        self.set_current_state(currentPosition=self.initial_positions, currentVelocities=self._currentVelocities, currentForce=self._currentForce, currentTemperature=self.temperature,
                               currentLambda=self._currentLam, currentdHdLam=self._currentdHdLam)

    def init_velocities(self)-> NoReturn:
        self._currentVelocities = np.array([[self._gen_rand_vel() for dim in range(self.nDim)] for state in range(self.nStates)] if(self.nDim>1) else [self._gen_rand_vel() for state in range(self.nStates)])
        self.veltemp = np.sum(self.mass / const.gas_constant / 1000.0 * np.linalg.norm(self._currentVelocities) ** 2) # t
        return self._currentVelocities

    def set_current_state(self, currentPosition:(Number or Iterable), currentLambda:(Number or Iterable),
                          currentVelocities:(Number or Iterable)=0,  currentdHdLam:(Number or Iterable)=0,
                          currentForce:(Number or Iterable)=0, currentTemperature:Number=298):
        self._currentPosition = currentPosition
        self._currentForce = currentForce
        self._currentVelocities = currentVelocities
        self._currentTemperature = currentTemperature

        self.updateEne()
        self.updateCurrentState()

    def updateCurrentState(self):
        self.currentState = self.state(position=self._currentPosition, temperature=self._currentTemperature,
                                       totEnergy=self._currentTotE,
                                       totPotEnergy=self._currentTotPot, totKinEnergy=self._currentTotKin,
                                       dhdpos=self._currentForce, velocity=self._currentVelocities,
                                       lam=self._currentLam, dhdlam=self._currentdHdLam)

    def append_state(self, newPosition, newVelocity, newForces, newLam):
        self._currentPosition = self.potential._check_positions_type_singlePos(newPosition)
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

    def totKin(self)-> (Iterable[Number] or Number or None):
      # Todo: more efficient if?
      if(self.nDim == 1 and isinstance(self._currentVelocities, Number) and not np.isnan(self._currentVelocities)):
          return 0.5 * self.mass * np.square(np.linalg.norm(self._currentVelocities))
      elif(self.nDim > 1 and isinstance(self._currentVelocities, Iterable) and all([isinstance(x, Number) and not np.isnan(x) for x in self._currentVelocities])):
          return np.sum(0.5 * self.mass * np.square(np.linalg.norm(self._currentVelocities)))
      else:
          return np.nan

    def totPot(self)-> (Iterable[Number] or Number or None):
      return self.potential.ene(self._currentPosition)

    def _update_dHdlambda(self):
        self._currentdHdLam = self.potential.dhdlam(self._currentPosition)
        self.updateCurrentState()
        return self._currentdHdLam
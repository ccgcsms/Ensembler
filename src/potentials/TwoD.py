

"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import numpy as np
import math
import typing as t

from Ensembler2.src.potentials import ND

class potential2DCls(ND._potentialNDCls):
    '''
    potential base class
    '''
    nDim:int =2


class wavePotential2D(potential2DCls):

    '''

        .. autoclass:: Wave Potential

    '''

    name:str = "Wave Potential"
    phase_shift:float = 0.0
    amplitude:float = 1.0
    multiplicity:float = 1.0
    radians:bool = False

    def __init__(self,  phase_shift:tuple=(0.0, 0.0), multiplicity:tuple=(1.0, 1.0), amplitude:tuple=(1.0, 1.0), radians:bool=False):

        '''

        initializes wavePotential potential class

        '''

        super().__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.set_radians(radians)

        if(radians):
            self.phase_shift = phase_shift
        else:
            self.phase_shift = np.deg2rad(phase_shift)

    def set_degrees(self, degrees:bool=True):
        self.set_radians(radians=not degrees)

    def set_radians(self, radians:bool=True):
        self.radians=radians

        if(radians):
            self._calculate_energies = lambda positions: list(map(lambda coords:
                          sum([self.amplitude[ind]*math.cos(self.multiplicity[ind]*(x + self.phase_shift[ind])) for ind,x in enumerate(coords)]), positions))

            self._calculate_dhdpos =  lambda positions: list(map(lambda coords:
                          sum([self.amplitude[ind]*math.sin(self.multiplicity[ind]*(x + self.phase_shift[ind])) for ind,x in enumerate(coords)]), positions))

        else:
            self._calculate_energies = lambda positions: list(map(lambda coords:
                          sum([self.amplitude[ind]*math.cos(self.multiplicity[ind]*(x + self.phase_shift[ind])) for ind,x in enumerate(np.deg2rad(coords))]), positions))

            self._calculate_dhdpos =  lambda positions: list(map(lambda coords:
                          sum([self.amplitude[ind]*math.sin(self.multiplicity[ind]*(x + self.phase_shift[ind])) for ind,x in enumerate(np.deg2rad(coords))]), positions))


class torsionPotential2D(potential2DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "Torsion Potential"

    phase:float=1.0
    wave_potentials:t.List[wavePotential2D]=[]

    def __init__(self, wave_potentials:t.List[wavePotential2D]):
        '''
        initializes torsions Potential
        '''
        super().__init__()
        if(type(wave_potentials) == list):
            self.wave_potentials=wave_potentials
        else:
            self.wave_potentials=[wave_potentials]

        if(len(self.wave_potentials)>1):
            self._calculate_energies = lambda positions: np.add(*map(lambda x: np.array(x.ene(positions)), self.wave_potentials))
            self._caclulcate_dhdpos = lambda positions: np.add(*map(lambda x: np.array(x.dhdpos(positions)), self.wave_potentials))
        elif(len(self.wave_potentials)==1):
            self._calculate_energies = lambda positions: np.array(self.wave_potentials[0].ene(positions))
            self._caclulcate_dhdpos = lambda positions: np.array(self.wave_potentials[0].dhdpos(positions))
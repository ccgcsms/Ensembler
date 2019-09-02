

"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import numpy as np
import numbers
import math
import typing as t

from Ensembler.src.potentials import ND

class _potential2DCls(ND._potentialNDCls):
    '''
    potential base class
    '''
    nDim:int =2

    def __init__(self):
        super().__init__()
        self.nDim:int =2

    @classmethod
    def _check_positions_type_singlePos(cls, position: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        positions = super()._check_positions_type_singlePos(position=position)
        if(all([len(pos) == cls.nDim for pos in positions])):
            return positions
        else:
            raise Exception("Dimensionality is not correct for positions! "+str(positions))

    @classmethod
    def _check_positions_type_multiPos(cls, positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        positions = super()._check_positions_type_multiPos(positions=positions)

        #dim check
        if(len(positions) == 1 and all([ isinstance(pos, t.Iterable) and len(pos) == cls.nDim  and all([isinstance(x, numbers.Number) for x in pos]) for pos in positions[0]])):
            return positions[0]
        elif(all([len(pos) == cls.nDim for pos in positions])):
            return positions
        else:
            raise Exception("Dimensionality is not correct for positions! "+str(positions))

class wavePotential(_potential2DCls):

    '''

        .. autoclass:: Wave Potential

    '''

    name:str = "Wave Potential"
    phase_shift:(np.array or list) = [0.0, 0.0]
    amplitude:(np.array or list) = [1.0 , 1.0]
    multiplicity:(np.array or list) = [1.0, 1.0]
    y_offset:(np.array or list)= (0.0, 0.0)
    radians:bool = False

    def __init__(self,  phase_shift:tuple=(0.0, 0.0), multiplicity:tuple=(1.0, 1.0), amplitude:tuple=(1.0, 1.0), y_offset:tuple=(0,0), radians:bool=False):

        '''

        initializes wavePotential potential class

        '''

        super().__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.set_radians(radians)
        self.y_offset = y_offset

        if(radians):
            if(isinstance(phase_shift, numbers.Number)):
                self.phase_shift = [phase_shift, phase_shift]
            else:
                self.phase_shift = phase_shift
        else:
            if(isinstance(phase_shift, numbers.Number)):
                self.phase_shift = np.deg2rad([phase_shift, phase_shift])
            else:
                self.phase_shift = np.deg2rad(phase_shift)

    def _calculate_energies_singlePos(self, position: np.array) -> float:
        return sum(list(map(lambda ind,coord: self.amplitude[ind]*math.cos(self.multiplicity[ind]*(coord + self.phase_shift[ind]))+self.y_offset[ind], range(2), position)))

    def _calculate_dhdpos_singlePos(self, position: np.array) -> np.array:
        return np.array(list(map(lambda ind, coord: self.amplitude[ind]*math.sin(self.multiplicity[ind]*(coord + self.phase_shift[ind]))+self.y_offset[ind], range(2),position)))

    def _set_singlePos_mode(self):
        self._singlePos_mode = True
        if(self.radians):
            self.set_radians()
        else:
            self.set_degrees()

    def _set_multiPos_mode(self):
        self._singlePos_mode = False
        if(self.radians):
            self.set_radians()
        else:
            self.set_degrees()

    def set_degrees(self, degrees: bool = True):
        self.radians = not degrees
        if (degrees):
            if (self._singlePos_mode):
                self._calculate_energies = lambda positions: self._calculate_energies_singlePos(np.deg2rad(positions))
                self._calculate_dhdpos = lambda positions: self._calculate_dhdpos_singlePos(np.deg2rad(positions))
            else:
                self._calculate_energies = lambda positions: self._calculate_energies_multiPos(np.deg2rad(positions))
                self._calculate_dhdpos = lambda positions: self._calculate_dhdpos_multiPos(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            if(self._singlePos_mode):
                self._calculate_energies  = self._calculate_energies_singlePos
                self._calculate_dhdpos = self._calculate_dhdpos_singlePos
            else:
                self._calculate_energies  = self._calculate_energies_multiPos
                self._calculate_dhdpos = self._calculate_dhdpos_multiPos
        else:
            self.set_degrees(degrees=not radians)


class torsionPotential(_potential2DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "Torsion Potential"

    phase:float=1.0
    wave_potentials:t.List[wavePotential]=[]

    def __init__(self, wave_potentials:t.List[wavePotential]):
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
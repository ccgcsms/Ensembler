"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""
import os

import numpy as np
import math
from numbers import Number
from typing import Iterable, List, Sized, Union

from Ensembler.src.potentials._baseclasses import _potential2DCls

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
            if(isinstance(phase_shift, Number)):
                self.phase_shift = [phase_shift, phase_shift]
            else:
                self.phase_shift = phase_shift
        else:
            if(isinstance(phase_shift, Number)):
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
                self._check_positions_type = self._check_positions_type_singlePos
                self._calculate_energies = lambda positions: self._calculate_energies_singlePos(np.deg2rad(positions))
                self._calculate_dhdpos = lambda positions: self._calculate_dhdpos_singlePos(np.deg2rad(positions))
            else:
                self._check_positions_type = self._check_positions_type_multiPos
                self._calculate_energies = lambda positions: self._calculate_energies_multiPos(np.deg2rad(positions))
                self._calculate_dhdpos = lambda positions: self._calculate_dhdpos_multiPos(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            if(self._singlePos_mode):
                self._check_positions_type = self._check_positions_type_singlePos
                self._calculate_energies  = self._calculate_energies_singlePos
                self._calculate_dhdpos = self._calculate_dhdpos_singlePos
            else:
                self._check_positions_type = self._check_positions_type_multiPos
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
    wave_potentials:List[wavePotential]=[]

    def __init__(self, wave_potentials:List[wavePotential]):
        '''
        initializes torsions Potential
        '''
        super().__init__()

        if(isinstance(wave_potentials, Sized) and len(wave_potentials) > 1):
            self.wave_potentials = wave_potentials
        else:
            raise Exception("Torsion Potential needs at least two Wave functions. Otherewise please use wave Potentials.")

    def _calculate_energies_multiPos(self, positions: Iterable[Iterable[Number]]) ->  np.array:
        return np.add(*map(lambda x: np.array(x.ene(positions)), self.wave_potentials))

    def _calculate_energies_singlePos(self, position: Iterable[float]) -> np.array:
        return np.add(*map(lambda x: np.array(x.ene(position)), self.wave_potentials))

    def _calculate_dhdpos_multiPos(self, positions: Iterable[Iterable[float]]) ->  np.array:
        return  np.add(*map(lambda x: np.array(x.dhdpos(positions)), self.wave_potentials))

    def _calculate_dhdpos_singlePos(self, position:Iterable[float]) -> np.array:
        return  np.add(*map(lambda x: np.array(x.dhdpos(position)), self.wave_potentials))


class swissPotential(_potential2DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "SwissPotential"
    look_up_table:np.array
    inverse:bool=False

    def __init__(self, inverse:bool=False):
        '''
        initializes torsions Potential
        '''
        super().__init__()

        #loading speedup!
        if(os.path.exists(os.path.dirname(__file__) + "/look_up_tables/Switzerland_20m/DTM_Switzerland_20m.npy")):
            self.look_up_table = np.load(  open(os.path.dirname(__file__) + "/look_up_tables/Switzerland_20m/DTM_Switzerland_20m.npy", "rb"))
        else:   #load intial from tif
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None

            self.look_up_table = np.array(Image.open(os.path.dirname(__file__) + "/look_up_tables/Switzerland_20m/DTM_Switzerland_20m.tif"))
            self.look_up_table = np.rot90(np.rot90(self.look_up_table))
            np.putmask(self.look_up_table, self.look_up_table == -32767, 32767)
            np.save( arr=self.look_up_table, file=os.path.dirname(__file__) + "/look_up_tables/Switzerland_20m/DTM_Switzerland_20m.npy")
        self.shape = self.look_up_table.shape
        self.set_Inverse(inverse)

    def set_Inverse(self, inverse:bool):
        if(inverse == self.inverse):
            pass
        else:
            self.inverse = inverse
            self.look_up_table = np.multiply(-1, self.look_up_table)
            np.putmask(self.look_up_table, self.look_up_table == -32767, 32767)

    def _calculate_energies_singlePos(self, position: Iterable[Number]) -> np.array:
        if(position[0]% 1 != 0 or position[0]% 1 != 0):
            #todo: interpolate!
            return self.look_up_table[int(round(position[0])), int(round(position[1]))]
        else:
            return self.look_up_table[position[0], position[1]]

    def _calculate_dhdpos_singlePos(self, position: Union[Iterable[Number], Number]) -> Union[np.array, Number]:
        self.look = lambda x,y: self.look_up_table(x,y)
        x = position[0]
        y = position[1]
        scan = [-1,0,1]
        for dy in scan:
            dhdpos_dim_sum = ((self.look(x, y)-self.look(x-1, y+dy))+
                          (self.look(x+1, y+dy)-self.look(x, y)))
        dhdpos_x = dhdpos_dim_sum/(len(scan)*2)

        for dx in scan:
            dhdpos_dim_sum = ((self.look(x, y)-self.look(x+dx, y-1))+
                          (self.look(x+dx, y+1)-self.look(x, y)))
        dhdpos_y = dhdpos_dim_sum/(len(scan)*2)

        return (dhdpos_x, dhdpos_y)

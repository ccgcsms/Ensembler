"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import math
import numbers
import numpy as np
from typing import Iterable, Union

from Ensembler.src.potentials._baseclasses import _potentialNDCls

"""
standard potentials
"""
class flat_well(_potentialNDCls):
    '''
        .. autoclass:: flat well potential
    '''
    name:str = "Flat Well"
    x_min: float = None
    x_max: float = None
    y_max:float = None
    y_min:float = None

    def __init__(self, x_range: list = [0, 1], y_max: float = 1000, y_min: float = 0):
        '''
        initializes flat well potential class

        '''
        super().__init__()

        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

    def _calculate_energies_singlePos(self, position: np.array) -> np.array:
        return np.array(list(map(lambda dimPos: self.y_min if (dimPos >= self.x_min and dimPos <= self.x_max) else self.y_max, position)))

    def _calculate_dhdpos_singlePos(self, position:(Iterable[float])) -> np.array:
        return np.zeros(shape=len(position))

class harmonicOsc(_potentialNDCls):
    '''
        .. autoclass:: harmonic oscillator potential
    '''
    name:str = "harmonicOscilator"
    x_shift = None
    fc = None

    def __init__(self, fc: float = 1.0, x_shift: float = 0.0, y_shift: float = 0.0):
        '''
        initializes harmonicOsc1D class
        :param fc: force constant
        :param x_shift: minimum position of harmonic oscillator
        '''
        super().__init__()
        self.fc = fc
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _calculate_energies_singlePos(self, position: np.array) -> np.array:
        return np.sum(np.array(list(map(lambda pos: 0.5 * self.fc * (pos - self.x_shift) ** 2 - self.y_shift, position))))

    def _calculate_dhdpos_singlePos(self, position: np.array) -> np.array:
        return np.array(list(map(lambda pos: self.fc * (pos - self.x_shift), position)))

"""
Waves
"""

class wavePotential(_potentialNDCls):
    '''
        .. autoclass:: Wave Potential
    '''
    name:str = "Wave Potential"

    phase_shift:float = 0.0
    amplitude:float = 1.0
    multiplicity:float = 1.0
    radians:bool = False

    def __init__(self, phase_shift: float = 0.0, multiplicity: float = 1.0, amplitude: float = 1.0,
                 radians: bool = False):
        '''
        initializes wavePotential potential class
        '''
        super().__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.set_radians(radians)
        if (radians):
            self.phase_shift = phase_shift
        else:
            self.phase_shift = np.deg2rad(phase_shift)

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
        self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            self._calculate_energies = lambda positions:  np.array(list(
                map(lambda dimPos: list(map(lambda x: self.amplitude * math.cos(self.multiplicity * (x + self.phase_shift)), dimPos)), positions)))
            self._calculate_dhdpos = lambda positions:  np.array(list(
                map(lambda dimPos: list(map(lambda x: self.amplitude * math.sin(self.multiplicity * (x + self.phase_shift)), dimPos)), positions)))
        else:
            self._calculate_energies = lambda positions: np.array(list(
                map(lambda dimPos: list(map(lambda x: self.amplitude * math.cos(self.multiplicity * (x + self.phase_shift)), dimPos)),
                    np.deg2rad(positions))))
            self._calculate_dhdpos = lambda positions: np.array(list(
                map(lambda dimPos: list(map(lambda x:self.amplitude * math.sin(self.multiplicity * (x + self.phase_shift)), dimPos)),
                    np.deg2rad(positions))))

"""
    ENVELOPED POTENTIALS
"""
class envelopedPotential(_potentialNDCls):
    """
    .. autoclass:: envelopedPotential
    """
    name = "Enveloping potential"
    V_is:Iterable[_potentialNDCls] = None
    E_is:Iterable[float] = None
    nStates:int = None
    s:float = None
    nStates:int = 0

    def __init__(self, V_is: Iterable[_potentialNDCls], s: float = 1.0, Eoff_i: Iterable[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """

        # Todo: think about n states with each m dims.
        if(self.nDim == -1):
            self.nDim = V_is[0].nDim

        super().__init__(nDim=self.nDim)

        #check State number
        self.nStates = len(V_is)
        #if (self.nStates < 2):
        #    raise IOError("It does not make sense enveloping less than two potentials!")
        if (Eoff_i == None):
            Eoff_i = [0.0 for state in range(len(V_is))]
        elif (len(Eoff_i) != self.nStates):
            raise IOError(
                "Energy offset Vector and state potentials don't have the same length!\n states in Eoff " + str(
                    len(Eoff_i)) + "\t states in Vi" + str(len(V_is)))

        if (any([V.nDim != self.nDim for V in V_is]) and not self.nDim == -1):
            raise Exception("Not all endstates have the same dimensionality! This is not imnplemented.\n Dims:\n\t envelopedPot: "+str(self.nDim)+"\n\t Potentials: " + str(
                [V.nDim != self.nDim for V in V_is]))

        self.V_is = V_is
        self.s = s
        self.Eoff_i = Eoff_i

    def __str__(self) ->str:
        msg = super().__str__()
        msg += "\tEnveloped Potentials: \n\t\t"+"\n\t\t".join(*[str(p).split("\n") for p in self.V_is])+"\n"
        return msg

    def _check_positions_type_singlePos(self, position: Union[Iterable[numbers.Number], numbers.Number]) -> np.array:
        #print(position)
        #print("hmmm", position)
        if (isinstance(position, numbers.Number)):
                return np.array([[position] for state in range(self.nStates)], ndmin=2)
        elif (isinstance(position, Iterable)):
            if(len(position) == self.nDim and all([isinstance(x, numbers.Number) for x in position])):    #ndim pot list
                return np.array([position for state in range(self.nStates)], ndmin=2)
            elif(len(position) == self.nDim and all([isinstance(x, Iterable) for x in position]) and all([isinstance(x, numbers.Number) for x in position for y in x ])):    #nDim pos lis
                return np.array([position for position in range(self.nStates)], ndmin=3)
            elif(len(position) == self.nStates):    #nDim pos lis
                if(all([isinstance(x, Iterable) and (len(x)==self.nDim or self.nDim==-1) and all([isinstance(y, numbers.Number) for y in x]) for x in position])):
                    return np.array(position, ndmin=2)
            #hack for indep. eds
            elif (len(position) == self.nStates and all([isinstance(x, Iterable) for x in position]) and len(position[0]) == self.nStates):
                return np.array([[x] for x in position[0]])

            else:
                raise Exception("This is an unknown type of Data structure: " + str(type(position)) + "\n" + str(position))
        else:
            raise Exception("This is an unknown type of Data structure: " + str(type(position)) + "\n" + str(position))

    def _check_positions_type_multiPos(self, positions: Union[Iterable[numbers.Number], numbers.Number]) -> np.array:
        if (isinstance(positions, numbers.Number)):
                return np.array([[positions] for state in range(self.nStates)], ndmin=3)
        elif (isinstance(positions, Iterable)):
            if(all([isinstance(x, numbers.Number) for x in positions]) and self.nDim ==1 ):    #ndim pot list
                return np.array([[[p] for state in range(self.nStates)] for p in positions], ndmin=3)
            elif(all([isinstance(x, numbers.Number) for x in positions])):  # ndim pot list
                return np.array([positions for state in range(self.nStates)], ndmin=3)
            elif(all([isinstance(x, Iterable) and (len(x) == self.nDim or self.nDim == -1) and all([isinstance(y, numbers.Number) for y in x]) for x in positions])):    #nDim pos lis
                return np.array([[pos for position in range(self.nStates)] for pos in positions])
            elif(all([isinstance(x, Iterable) and (len(x) == self.nStates or self.nDim == -1) and all([isinstance(y, numbers.Number) for y in x]) for x in positions])):    #nDim pos lis
                print("Don't be a maybe")
                return np.array([[pos] for pos in positions])
            elif(all([(isinstance(x, Iterable) and len(x) == self.nStates) and all([isinstance(y, Iterable) and (len(y) == self.nDim or self.nDim==-1) and all([isinstance(z, numbers.Number)  for z in y] ) for y in x]) for x in positions])):
                return np.array(positions, ndmin=3)
            else:
                raise Exception("This is an unknown type of Data structure, wrapped by a Iterable: " + str(type(positions)) + "\n" + str(positions))
        else:
            raise Exception("This is an unknown type of Data structure: " + str(type(positions)) + "\n" + str(positions))

    def _calculate_energies_singlePos(self, position:(Iterable[float])) -> np.array:
        #print("NDSi", position)

        partA = np.multiply(-self.s, np.subtract(self.V_is[0].ene(position[0]), self.Eoff_i[0]))
        if(len(self.Eoff_i) >1):
            partB = np.multiply(-self.s, np.subtract(self.V_is[1].ene(position[1]), self.Eoff_i[1]))
            sum_prefactors = max(partA, partB) + np.log(1 + np.exp(min(partA, partB) - max(partA, partB)))
        else:
            sum_prefactors = partA + np.log(1 + np.exp(partA - partA))
        #print("partA", partA)
        #print("partB", partB)


        # more than two states!
        for state in range(2, self.nStates):
            partN = np.multiply(-self.s, np.subtract(self.V_is[state].ene(position[state]), self.Eoff_i[state]))
            sum_prefactors = max(sum_prefactors, partN) + np.log(1 + np.exp(min(sum_prefactors, partN) - max(sum_prefactors, partN)))

        #print(sum_prefactors)
        Vr = float(np.sum(np.multiply(np.divide(-1, float(self.s)), sum_prefactors)))
        return Vr

    def _calculate_dhdpos_singlePos(self, position:(Iterable[float])) -> np.array:
        """
        :warning : Implementation is not entirly correct!
        :param position:
        :return:
        """
        ###CHECK!THIS FUNC!!! not correct
        V_R_ene = self.ene(position)
        V_Is_ene = np.array([statePot.ene(state_pos) for statePot, state_pos in zip(self.V_is, position)])
        V_Is_dhdpos = np.array([statePot.dhdpos(state_pos) for statePot, state_pos in zip(self.V_is, position)])
        dhdpos = []


        #print("POS: " , position.shape,"\n\t", position,)
        #print("ene: ", V_Is_ene.shape,"\n\t", V_Is_ene)
        #print("dhdpos: ", V_Is_dhdpos.shape,"\n\t", V_Is_dhdpos)
        #print("T", V_Is_ene.T)
        V_Is_posDim_eneSum = np.sum(V_Is_ene.T, axis=1).T
        #print("sums: ", V_Is_posDim_eneSum.shape, "\n\t", V_Is_posDim_eneSum)

        #prefactors = np.array([np.zeros(len(position[0])) for x in range(len(position))])
        #todo: error this should be ref pot fun not sum of all pots

        #print(position)
        prefactors = np.array([list(map(lambda dimPos: 1 - np.divide(dimPos, V_Is_posDim_eneSum), list(Vn_ene))) for Vn_ene in V_Is_ene])
        ##print("preFactors: ",prefactors.shape, "\n\t", prefactors,  "\n\t", prefactors.T)
        dhdpos_state_scaled = np.multiply(prefactors, V_Is_dhdpos)
        #print("dhdpos_scaled", dhdpos_state_scaled.shape, "\n\t", dhdpos_state_scaled, "\n\t", dhdpos_state_scaled.T    )

        #dhdpos_R = [  for dhdpos_state in dhdpos_state_scaled]

        dhdpos_R = []
        #print("Ndim: ", self.nDim)
        for dimPos in range(self.nDim):
            dhdposR_positionDim = 0
            for state in range(len(V_Is_ene)):
                dhdposR_positionDim = np.add(dhdposR_positionDim, dhdpos_state_scaled[state, dimPos])
            dlist = [dhdposR_positionDim]
            dlist.extend(dhdpos_state_scaled[:, dimPos])
            dhdpos_R.extend(dlist)
        return np.array(dhdpos_R)

    def _set_singlePos_mode(self):
        """
        ..autofunction :: _set_singlePos_mode

        :return:  -
        """
        self._singlePos_mode = True
        self._check_positions_type = self._check_positions_type_singlePos
        self._calculate_energies = self._calculate_energies_singlePos
        self._calculate_dhdpos = self._calculate_dhdpos_singlePos
        [V._set_singlePos_mode() for V in self.V_is]

    def _set_multiPos_mode(self):
        """
        ..autofunction :: _set_multiPos_mode

        :return:  -
        """
        super()._set_multiPos_mode()
        self._check_positions_type = self._check_positions_type_multiPos
        self._calculate_energies = self._calculate_energies_multiPos
        self._calculate_dhdpos = self._calculate_dhdpos_multiPos
        [V._set_multiPos_mode() for V in self.V_is]

    def _set_no_type_check(self):
        """
        ..autofunction :: _set_no_type_check
            This function is trying to speed up execution for cases, in that the position Type is known to be correct (system integration) ...

        :return:  -
        """
        super()._set_no_type_check()
        [V._set_no_type_check() for V in self.V_is]

    def _set_type_check(self):
        """
        ..autofunction :: _set_type_check
            This function is setting the default potential Value, to allow secure execution in small code snippets
        :return:  -
        """
        super()._set_type_check()
        [V._set_type_check() for V in self.V_is]

class envelopedPotentialMultiS(envelopedPotential):
    """
    .. autoclass:: envelopedPotential
    """
    V_is:Iterable[_potentialNDCls] = None
    E_is:Iterable[float] = None
    nStates:int = None
    s:Iterable[float] = None

    def __init__(self, V_is: Iterable[_potentialNDCls], s: Iterable[float], Eoff_i: Iterable[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        super().__init__(V_is=V_is, Eoff_i=Eoff_i)
        self.s = s


    def _calculate_energies_singlePos(self, position:(Iterable[float])) -> np.array:
        #print("NDSi", position)

        partA = np.multiply(-self.s[0], np.subtract(self.V_is[0].ene(position[0]), self.Eoff_i[0]))
        partB = np.multiply(-self.s[1], np.subtract(self.V_is[1].ene(position[1]), self.Eoff_i[1]))
        #print("partA", partA)
        #print("partB", partB)

        sum_prefactors = max(partA, partB) + np.log(1 + np.exp(min(partA, partB) - max(partA, partB)))

        # more than two states!
        for state in range(2, self.nStates):
            partN = np.multiply(-self.s[state], np.subtract(self.V_is[state].ene(position[state]), self.Eoff_i[state]))
            sum_prefactors = max(sum_prefactors, partN) + np.log(1 + np.exp(min(sum_prefactors, partN) - max(sum_prefactors, partN)))

        #print(sum_prefactors)
        Vr = float(np.sum(np.multiply(-1, sum_prefactors)))
        return Vr

    
    
    
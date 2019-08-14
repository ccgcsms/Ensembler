"""
Module: Potential
    This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import math
import numbers
import numpy as np
import typing as t
from collections import Iterable

class _potentialNDCls:
    '''
    potential base class
    '''
    name:str = "Unknown"
    nDim:int =-1

    def __init__(self, *kargs):
        return

    def __name__(self)->str:
        return str(self.name)

    @classmethod
    def _check_positions_type(cls, positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> t.List[float]:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """

        if(isinstance(positions, Iterable) and all([isinstance(dimPos, Iterable) and all([isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in positions])):
            return np.array(positions)
        elif(isinstance(positions, numbers.Number)):
            return np.array([[positions]])
        elif (isinstance(positions, Iterable) and all([isinstance(x, numbers.Number) for x in positions])):
            return np.array([positions])
        else:
            raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=" + str(
                len(positions)))

    def _calculate_energies(self, positions:t.List[float], *kargs):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos(self, positions:t.List[float], *kargs):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def ene(self, positions:(t.List[float] or float), *kargs) -> (t.List[float] or float):
        '''
        calculates energy of particle
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: energy
        '''

        positions = self._check_positions_type(positions)
        return self._calculate_energies(positions)

    def dhdpos(self, positions:(t.List[float] or float), *kargs) -> (t.List[float] or float):
        '''
        calculates derivative with respect to position
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdpos(positions)



"""
standard potentials
"""
class flat_wellND(_potentialNDCls):
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
        super(_potentialNDCls, self).__init__()
        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

    def _calculate_energies(self, positions, *kargs):
        return np.array([np.sum(list(map( lambda pos: self.y_min if (pos >= self.x_min and pos <= self.x_max) else self.y_max, dimPos))) for dimPos in positions])

    def _calculate_dhdpos(self, positions: (t.List[float] or float), *kargs) -> (t.List[float] or float):
        return np.zeros(shape=len(positions))


class harmonicOscND(_potentialNDCls):
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
        super(_potentialNDCls, self).__init__()
        self.fc = fc
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _calculate_energies(self, positions: t.List[float], *kargs) -> (t.List[float]):
        return np.array([np.sum(list(map(lambda pos: 0.5 * self.fc * (pos - self.x_shift) ** 2 - self.y_shift, dimPos))) for dimPos in
                positions])

    def _calculate_dhdpos(self, positions: t.List[float], *kargs) -> (t.List[float]):
        return np.array([np.sum(list(map(lambda pos: self.fc * (pos - self.x_shift), dimPos))) for dimPos in positions])

"""
Waves
"""

class wavePotential1D(_potentialNDCls):
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
        super(_potentialNDCls).__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.set_radians(radians)
        if (radians):
            self.phase_shift = phase_shift
        else:
            self.phase_shift = np.deg2rad(phase_shift)

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
    V_is:t.List[_potentialNDCls] = None
    E_is:t.List[float] = None
    numStates:int = None
    s:float = None

    def __init__(self, V_is: t.List[_potentialNDCls], s: float = 1.0, Eoff_i: t.List[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        super(_potentialNDCls).__init__()
        self.numStates = len(V_is)
        if (self.numStates < 2):
            raise IOError("It does not make sense enveloping less than two potentials!")
        if (Eoff_i == None):
            Eoff_i = [0.0 for state in range(len(V_is))]
        elif (len(Eoff_i) != self.numStates):
            raise IOError(
                "Energy offset Vector and state potentials don't have the same length!\n states in Eoff " + str(
                    len(Eoff_i)) + "\t states in Vi" + str(len(V_is)))

        # Todo: think about n states with each m dims.
        self.nDim = V_is[0].nDim
        if (any([V.nDim != self.nDim for V in V_is])):
            raise Exception("Not all endstates have the same dimensionality! This is not imnplemented.\n Dims: " + str(
                [V.nDim != self.nDim for V in V_is]))

        self.V_is = V_is
        self.s = s
        self.Eoff_i = Eoff_i

    # each state gets a position list
    def _check_positions_type(self, positions: t.List[float]) -> t.List[float]:
        if (type(positions) in [float, int, str]):
            if (len(positions) != self.numStates):
                positions = [positions for state in range(self.numStates + 1)]
            else:
                positions = [float(positions) for state in range(self.numStates + 1)]
        elif (isinstance(positions, Iterable)):
            if (len(positions) != self.numStates):
                if (isinstance(positions[0], Iterable)):  # Ndimensional case
                    positions = [list(map(lambda dimlist: np.array(list(map(float, dimlist))), positions)) for state in
                                 range(self.numStates)]
                else:  # onedimensional
                    positions = [list(map(float, positions)) for state in range(self.numStates)]
            else:  # TODO: insert check here! for fitting numstates
                return positions
        else:
            raise Exception(
                "This is an unknown type of Data structure: " + str(type(positions)) + "\n" + str(positions))
        return positions

    def _calculate_energies(self, positions: (t.List[float] or float), *kargs) -> list:
        partA = [-self.s * (Vit - self.Eoff_i[0]) for Vit in self.V_is[0].ene(positions[0])]
        partB = [-self.s * (Vit - self.Eoff_i[1]) for Vit in self.V_is[1].ene(positions[1])]
        sum_prefactors = [max(A_t, B_t) + math.log(1 + math.exp(min(A_t, B_t) - max(A_t, B_t))) for A_t, B_t in
                          zip(partA, partB)]

        # more than two states!
        for state in range(2, self.numStates):
            partN = [-self.s * (Vit - self.Eoff_i[state]) for Vit in self.V_is[state].ene(positions[state])]
            sum_prefactors = [max(sum_prefactors_t, N_t) + math.log(
                1 + math.exp(min(sum_prefactors_t, N_t) - max(sum_prefactors_t, N_t))) for sum_prefactors_t, N_t in
                              zip(sum_prefactors, partN)]

        Vr = [-1 / float(self.s) * partitionF for partitionF in sum_prefactors]
        return Vr

    def _calculate_dhdpos(self, positions: (t.List[float] or float), *kargs):
        ###CHECK!THIS FUNC!!! not correct
        V_R_ene = self.ene(positions)
        V_Is_ene = [statePot.ene(state_pos) for statePot, state_pos in zip(self.V_is, positions)]
        V_Is_dhdpos = [statePot.dhdpos(state_pos) for statePot, state_pos in zip(self.V_is, positions)]
        dhdpos = []

        for position in range(len(positions[0])):
            dhdpos_R = 0
            dhdpos_state = []
            for V_state_ene, V_state_dhdpos in zip(V_Is_ene, V_Is_dhdpos):
                # den = sum([math.exp(-const.k *Vstate[position]) for Vstate in V_Is_ene])
                # prefactor = (math.exp(-const.k *V_state_ene[position]))/den if (den!=0) else 0
                if (V_state_ene[position] == 0):
                    prefactor = 0
                else:
                    prefactor = 1 - (V_state_ene[position] / (sum([Vstate[position] for Vstate in V_Is_ene]))) if (
                                sum([Vstate[position] for Vstate in V_Is_ene]) != 0) else 0
                # print(round(positions[0][position],2),"\t",round(prefactor,2),"\t" , round(V_state_dhdpos[position]), "\t", round(V_R_ene[position]))
                dhdpos_state.append(prefactor * V_state_dhdpos[position])
                dhdpos_R += prefactor * V_state_dhdpos[position]
                dlist = [dhdpos_R]
                dlist.extend(dhdpos_state)
            dhdpos.append(dlist)
        return dhdpos

class envelopedPotentialMultiS(envelopedPotential):
    """
    .. autoclass:: envelopedPotential
    """
    V_is:t.List[_potentialNDCls] = None
    E_is:t.List[float] = None
    numStates:int = None
    s:t.List[float] = None

    def __init__(self, V_is: t.List[_potentialNDCls], s: t.List[float], Eoff_i: t.List[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        super(_potentialNDCls).__init__()
        self.numStates = len(V_is)
        if (self.numStates < 2):
            raise IOError("It does not make sense enveloping less than two potentials!")
        if (Eoff_i == None):
            Eoff_i = [0.0 for state in range(len(V_is))]

        elif (len(Eoff_i) != self.numStates):
            raise IOError(
                "Energy offset Vector and state potentials don't have the same length!\n states in Eoff " + str(
                    len(Eoff_i)) + "\t states in Vi" + str(len(V_is)))

        # Todo: think about n states with each m dims.
        self.nDim = V_is[0].nDim
        if (any([V.nDim != self.nDim for V in V_is])):
            raise Exception("Not all endstates have the same dimensionality! This is not imnplemented.\n Dims: " + str(
                [V.nDim != self.nDim for V in V_is]))

        self.V_is = V_is
        self.s = s
        self.Eoff_i = Eoff_i

    def _calculate_energies(self, positions: (t.List[float] or float), *kargs) -> list:
        partA = [-self.s[0] * (Vit - self.Eoff_i[0]) for Vit in self.V_is[0].ene(positions[0])]
        partB = [-self.s[1] * (Vit - self.Eoff_i[1]) for Vit in self.V_is[1].ene(positions[1])]
        sum_prefactors = [max(A_t, B_t) + math.log(1 + math.exp(min(A_t, B_t) - max(A_t, B_t))) for A_t, B_t in
                          zip(partA, partB)]

        # more than two states!
        for state in range(2, self.numStates):
            partN = [-self.s[state] * (Vit - self.Eoff_i[state]) for Vit in self.V_is[state].ene(positions[state])]
            sum_prefactors = [max(sum_prefactors_t, N_t) + math.log(
                1 + math.exp(min(sum_prefactors_t, N_t) - max(sum_prefactors_t, N_t))) for sum_prefactors_t, N_t in
                              zip(sum_prefactors, partN)]

        Vr = [- partitionF for state, partitionF in enumerate(sum_prefactors)]  # 1/ float(self.s[0]) *
        return Vr

    def _calculate_dhdpos(self, positions: (t.List[float] or float), *kargs):
        ###CHECK!THIS FUNC!!! not correct
        V_R_ene = self.ene(positions)
        V_Is_ene = [statePot.ene(state_pos) for statePot, state_pos in zip(self.V_is, positions)]
        V_Is_dhdpos = [statePot.dhdpos(state_pos) for statePot, state_pos in zip(self.V_is, positions)]
        dhdpos = []

        for position in range(len(positions[0])):
            dhdpos_R = 0
            dhdpos_state = []
            for V_state_ene, V_state_dhdpos in zip(V_Is_ene, V_Is_dhdpos):
                # den = sum([math.exp(-const.k *Vstate[position]) for Vstate in V_Is_ene])
                # prefactor = (math.exp(-const.k *V_state_ene[position]))/den if (den!=0) else 0
                if (V_state_ene[position] == 0):
                    prefactor = 0
                else:
                    prefactor = 1 - (V_state_ene[position] / (sum([Vstate[position] for Vstate in V_Is_ene]))) if (
                            sum([Vstate[position] for Vstate in V_Is_ene]) != 0) else 0
                # print(round(positions[0][position],2),"\t",round(prefactor,2),"\t" , round(V_state_dhdpos[position]), "\t", round(V_R_ene[position]))
                dhdpos_state.append(prefactor * V_state_dhdpos[position])
                dhdpos_R += prefactor * V_state_dhdpos[position]
                dlist = [dhdpos_R]
                dlist.extend(dhdpos_state)
            dhdpos.append(dlist)
        return dhdpos


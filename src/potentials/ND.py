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
    nDim:int =0

    def __init__(self):
        return

    def __name__(self)->str:
        return str(self.name)

    @classmethod
    def _check_positions_type(cls, positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """

        #array
        if(isinstance(positions, Iterable) and all([isinstance(dimPos, Iterable) and (all([len(x) == cls.nDim for x in positions]) or cls.nDim == 0) and all([isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in positions])
            or isinstance(positions, Iterable) and (len(positions) == cls.nDim  or cls.nDim == 0) and all([isinstance(x, numbers.Number) for x in positions])
            or isinstance(positions, numbers.Number)):
            return np.array(positions, ndmin=2)
        else:
            raise Exception("list dimensionality does not fit to potential dimensionality! Input: " + str(positions))

    def _calculate_energies(self, positions:t.List[float]):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos(self, positions:t.List[float]):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def ene(self, positions:(t.List[float] or float)) -> (t.List[float] or float):
        '''
        calculates energy of particle
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: energy
        '''

        positions = self._check_positions_type(positions)
        return self._calculate_energies(positions)

    def dhdpos(self, positions:(t.List[float] or float)) -> (t.List[float] or float):
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

    def _calculate_energies(self, positions):
        return np.array([list(map(lambda pos: self.y_min if (pos >= self.x_min and pos <= self.x_max) else self.y_max, dimPos)) for dimPos in positions])

    def _calculate_dhdpos(self, positions: (t.List[float] or float)) -> (t.List[float] or float):

        return np.array([np.zeros(shape=len(positions[0])) for x in range(len(positions))], ndmin=2)


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

    def _calculate_energies(self, positions: t.List[float]) -> (t.List[float]):
        return np.array([list(map(lambda pos: 0.5 * self.fc * (pos - self.x_shift) ** 2 - self.y_shift, dimPos)) for dimPos in
                positions])

    def _calculate_dhdpos(self, positions: t.List[float]) -> (t.List[float]):
        return np.array([list(map(lambda pos: self.fc * (pos - self.x_shift), dimPos)) for dimPos in positions])

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
        super().__init__()
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
    def _check_positions_type(self, positions: t.List[float]) -> np.array:
        if (type(positions) in [float, int, str]):
            if (len(positions) != self.numStates):
                return np.array([positions for state in range(self.numStates + 1)], ndmin=3)
            else:
                return np.array([float(positions) for state in range(self.numStates + 1)], ndmin=3)
        elif (isinstance(positions, Iterable)):
            if (len(positions) != self.numStates):
                if (isinstance(positions[0], Iterable) and all([ isinstance(dimPos, numbers.Number) for pos in positions for dimPos in pos])):  # Ndimensional case
                    return np.array([positions for state in range(self.numStates)], ndmin=3)
                else:  # onedimensional
                    return np.array([positions for state in range(self.numStates)], ndmin=3)
            else:  # TODO: insert check here! for fitting numstates
                return np.array(positions, ndmin=3)
        else:
            raise Exception(
                "This is an unknown type of Data structure: " + str(type(positions)) + "\n" + str(positions))

    def _calculate_energies(self, positions: (t.List[float] or float)) -> list:
        print(positions.shape)
        partA = np.array([-self.s * (Vit - self.Eoff_i[0]) for Vit in self.V_is[0].ene(positions[0])])
        partB = np.array([-self.s * (Vit - self.Eoff_i[1]) for Vit in self.V_is[1].ene(positions[1])])
        sum_prefactors = np.array([list(map(lambda A_t, B_t: max(A_t, B_t) + math.log(1 + math.exp(min(A_t, B_t) - max(A_t, B_t))), A, B)) for A, B in
                          zip(partA, partB)])

        # more than two states!
        for state in range(2, self.numStates):
            partN = [-self.s * (Vit - self.Eoff_i[state]) for Vit in self.V_is[state].ene(positions[state])]
            sum_prefactors = np.array(
                [list(map(lambda A_t, B_t: max(A_t, B_t) + math.log(1 + math.exp(min(A_t, B_t) - max(A_t, B_t))), A, B))
                 for A, B in
                 zip(sum_prefactors, partN)])

        Vr = [-1 / float(self.s) * partitionF for partitionF in sum_prefactors]
        return np.array(Vr)

    def _calculate_dhdpos(self, positions: (t.List[float] or float)):
        """
        :warning : Implementation is not entirly correct!
        :param positions:
        :return:
        """
        ###CHECK!THIS FUNC!!! not correct
        V_R_ene = self.ene(positions)
        V_Is_ene = np.array([statePot.ene(state_pos) for statePot, state_pos in zip(self.V_is, positions)])
        V_Is_dhdpos = np.array([statePot.dhdpos(state_pos) for statePot, state_pos in zip(self.V_is, positions)])
        dhdpos = []

        #print("POS: " , positions.shape,"\n\t", positions,)
        #print("ene: ", V_Is_ene.shape,"\n\t", V_Is_ene)
        #print("dhdpos: ", V_Is_dhdpos.shape,"\n\t", V_Is_dhdpos)
        #print("T", V_Is_ene.T)
        V_Is_posDim_eneSum = np.sum(V_Is_ene.T, axis=2).T
        #print("sums: ", V_Is_posDim_eneSum.shape, "\n\t", V_Is_posDim_eneSum)

        #prefactors = np.array([np.zeros(len(positions[0])) for x in range(len(positions))])
        #todo: error this should be ref pot fun not sum of all pots

        prefactors = np.array([list(map(lambda pos, posSum: list(map(lambda dimPos, dimPosSum: 1 - np.divide(dimPos, dimPosSum), pos, posSum)), Vn_ene, V_Is_posDim_eneSum)) for Vn_ene in V_Is_ene])
        ##print("preFactors: ",prefactors.shape, "\n\t", prefactors,  "\n\t", prefactors.T)
        dhdpos_state_scaled = np.multiply(prefactors, V_Is_dhdpos)
        #print("dhdpos_scaled", dhdpos_state_scaled.shape, "\n\t", dhdpos_state_scaled, "\n\t", dhdpos_state_scaled.T    )

        #dhdpos_R = [  for dhdpos_state in dhdpos_state_scaled]

        dhdpos_R = []
        for position in range(len(positions[0])):
            dhdposR_position = []
            for dimPos in range(len(positions[0][0])):
                dhdposR_positionDim = 0
                for state in range(len(V_Is_ene)):
                    dhdposR_positionDim = np.add(dhdposR_positionDim, dhdpos_state_scaled[state, position, dimPos])
                dlist = [dhdposR_positionDim]
                dlist.extend(dhdpos_state_scaled[:, position, dimPos])
                dhdposR_position.append(dlist)
            dhdpos_R.append(dhdposR_position)

        return np.array(dhdpos_R)

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

    def _calculate_energies(self, positions: (t.List[float] or float)) -> list:
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

    def _calculate_dhdpos(self, positions: (t.List[float] or float)):
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


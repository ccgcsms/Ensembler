"""
Module: Potential
This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import math
import numpy as np
import typing as t
import numbers
import scipy.constants as const
from collections.abc import Iterable, Sized

from Ensembler.src.potentials import ND

class _potential1DCls(ND._potentialNDCls):
    '''
        .. autoclass:: _potentialCls
        This class is the
    '''

    nDim:int = 1

    def __init__(self):
        super().__init__(nDim=self.nDim)

    @classmethod
    def _check_positions_type_singlePos(cls, position: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param position: here the positions that shall be evaluated by the potential should be given.
        :type position:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """
        if (isinstance(position, numbers.Number)):
            return np.float(position)
        elif(isinstance(position, Sized) and len(position) == cls.nDim and isinstance(position[0], numbers.Number)):
            return np.float(position[0])
        else:
            raise Exception("Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))

    @classmethod
    def _check_positions_type_multiPos(cls,
                                        positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """
        if (isinstance(positions, numbers.Number)):  # single number
            return np.array(positions, ndmin=1)
        elif (isinstance(positions, Iterable)):
            if (all([isinstance(x, numbers.Number) for x in positions])):  # list with numbers
                return np.array(positions, ndmin=1)
            elif ((type(positions) != type(None) and len(positions) == 1) and all(
                    [isinstance(pos, numbers.Number) for pos in positions[0]])):
                return np.array(positions[0], ndmin=1)
            else:
                raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=2 potential Dimensions 1")
        else:
            if (type(positions) == type(None)):
                raise Exception("potential got None as position")
            else:
                raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=" + str(
                    len(positions)) + " potential Dimensions " + str(cls.nDim))

    def _calculate_energies_singlePos(self, position: float)  -> float:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos_singlePos(self, position: float)  -> float:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

"""
    SIMPLE POTENTIALS
"""

class flat_well(_potential1DCls):
    '''
        .. autoclass:: flat well potential
    '''
    name:str = "Flat Well"
    x_min: float = None
    x_max: float = None
    y_max:float = None
    y_min:float = None

    def __init__(self, x_range: list = (0, 1), y_max: float = 1000, y_min: float = 0):
        '''
        initializes flat well potential class

        '''
        super().__init__()
        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.y_min if (position >= self.x_min and position <= self.x_max) else self.y_max

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return 0

class harmonicOsc(_potential1DCls):
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
        self.nDim=1
        self.fc = fc
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _calculate_energies_singlePos(self, position:float) ->  float:
        return 0.5 * self.fc * (position - self.x_shift) ** 2 - self.y_shift

    def _calculate_dhdpos_singlePos(self, position:float) ->  float:
        return self.fc * (position - self.x_shift)

class wavePotential(_potential1DCls):
    '''
        .. autoclass:: Wave Potential
    '''
    name:str = "Wave Potential"

    phase_shift:float = 0.0
    amplitude:float = 1.0
    multiplicity:float = 1.0
    y_offset:float = 0.0
    radians:bool = False

    def __init__(self, phase_shift: float = 0.0, multiplicity: float = 1.0, amplitude: float = 1.0, y_offset:float = 0.0,
                 radians: bool = False):
        '''
        initializes wavePotential potential class
        '''
        super().__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.y_offset = y_offset
        self.set_radians(radians)

        if (radians):
            self.phase_shift = phase_shift
        else:
            self.phase_shift = np.deg2rad(phase_shift)

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.amplitude * math.cos(self.multiplicity * (position + self.phase_shift)) + self.y_offset

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return self.amplitude * math.sin(self.multiplicity * (position + self.phase_shift)) + self.y_offset

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
        if(degrees):
            self._calculate_energies = lambda positions: self._calculate_energies_multiPos(np.deg2rad(positions))
            self._calculate_dhdpos = lambda positions: self._calculate_dhdpos_multiPos(np.deg2rad(positions))
        else:
            self.set_radians(radians=not degrees)

    def set_radians(self, radians: bool = True):
        self.radians = radians
        if (radians):
            self._calculate_energies  =self._calculate_energies_multiPos
            self._calculate_dhdpos = self._calculate_dhdpos_multiPos
        else:
            self.set_degrees(degrees=not radians)

class torsionPotential(_potential1DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "Torsion Potential"

    phase:float = 1.0
    wave_potentials:t.Iterable[wavePotential] = []

    def __init__(self, wave_potentials: t.Union[t.Iterable[wavePotential], wavePotential]):
        '''
        initializes torsions Potential
        '''
        super().__init__()
        if (isinstance(wave_potentials, Iterable) and len(wave_potentials) >1):
            self.wave_potentials = wave_potentials
            self._calculate_energies = lambda positions: np.add(
                *map(lambda x: np.array(x.ene(positions)), self.wave_potentials))
            self._calculate_dhdpos = lambda positions: np.add(
                *map(lambda x: np.array(x.dhdpos(positions)), self.wave_potentials))
        elif (isinstance(wave_potentials, Iterable) and len(wave_potentials) == 1):
            self.wave_potentials = wave_potentials[0]
            self._calculate_energies = lambda positions: self.wave_potentials.ene(positions)
            self._calculate_dhdpos = lambda positions: self.wave_potentials.dhdpos(positions)
        elif(isinstance(wave_potentials, wavePotential)):
            self.wave_potentials = wave_potentials
            self._calculate_energies = lambda positions: self.wave_potentials.ene(positions)
            self._calculate_dhdpos = lambda positions: self.wave_potentials.dhdpos(positions)
        else:
            raise IOError("the provided potential was not recognizable.")

    def _set_singlePos_mode(self):
        """
        ..autofunction :: _set_singlePos_mode
            Functionality not implemented for EDSPOT
        :return:  -
        """
        pass
    def _set_multiPos_mode(self):
        """
        ..autofunction :: _set_multiPos_mode
            Functionality not implemented for EDSPOT
        :return:  -
        """
        pass

class coulombPotential(_potential1DCls):
    name = "Coulomb Potential"

    epsilon:float

    coulombLaw = lambda t, q1, q2, r, epsilon: np.divide(np.multiply(q1, q2), np.multiply(r, epsilon * 4 * math.pi))
    dcoulombLawdr = lambda t, q1, q2, r, epsilon: np.divide(np.multiply(-q1, q2), np.multiply(np.square(r), epsilon * 4 * math.pi))

    def __init__(self, q1=1, q2=1, epsilon=1):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.epsilon = epsilon
        pass

    def _calculate_energies_singlePos(self, distance: float) -> float:
        return self.coulombLaw(self.q1, self.q2, distance, self.epsilon)

    def _calculate_dhdpos_singlePos(self, distance: float) -> float:
        return self.dcoulombLawdr(self.q1, self.q2, distance, self.epsilon)

class lennardJonesPotential(_potential1DCls):
    '''
        .. autoclass:: Lennard Jones Potential
    '''
    name:str = "Lennard Jones Potential"

    c6: float = None
    c12: float = None
    x_shift: float = None
    y_shift: float = None

    def __init__(self, c6: float = 0.2, c12: float = 0.0001, x_shift: float = 0, y_shift=0):
        '''
        initializes flat well potential class
        '''
        super().__init__()
        self.c6 = c6
        self.c12 = c12
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _calculate_energies_singlePos(self, position: float) -> float:
        return np.subtract(np.divide(self.c12, np.power(position-self.x_shift, 12)), np.divide(self.c6, np.power(position-self.x_shift, 6)))+self.y_shift

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return 6 * ((2 * self.c12) - ((position -self.x_shift)** 6 * self.c6)) / (position-self.x_shift) ** 13

class doubleWellPot(_potential1DCls):
    '''
        .. autoclass:: unperturbed double well potential
    '''
    name:str = "Double Well Potential"

    a = None
    b = None
    Vmax = None

    def __init__(self, Vmax=100.0, a=0.0, b=0.5):
        '''
        initializes double well potential
        :param Vmax:
        :param a:
        :param b:
        '''
        super().__init__()
        self.Vmax = Vmax
        self.a = a
        self.b = b

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.Vmax / self.b ** 4 * ((position - self.a / 2) ** 2 - self.b ** 2) ** 2

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return 4 * self.Vmax / self.b ** 4 * ((position - self.a / 2) ** 2 - self.b ** 2) * (position - self.a / 2)


"""
    PERTURBED POTENTIALS
"""
class _perturbedPotential1DCls(_potential1DCls):
    """
        .. autoclass:: perturbedPotentialCls
    """
    nStates = 2
    lam:float

    def __init__(self, lam: float = 0.0):
        '''
        Initializes a potential of the form V = 0.5 * (1 + alpha * lam) * fc * (pos - gamma * lam) ** 2
        :param fc: force constant
        :param alpha: perturbation parameter for width of harmonic oscillator
        :param gamma: perturbation parameter for position of harmonic oscillator
        '''
        super().__init__()
        self.lam = lam
        self._calculate_dhdlam = self._calculate_dhdlam_multiPos

    def _calculate_dhdlam(self, positions: t.List[float], lam: float = 1.0):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def set_lam(self, lam: float):
        self.lam = lam

    def dhdlam(self, positions: (t.List[float] or float)) -> (t.List[float] or float):
        '''
        calculates derivative with respect to lambda value
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dlan
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdlam(positions)

    def _calculate_dhdlam_singlePos(self, positoin:float) -> (np.array or float):
        raise Exception("Please implement this function!")

    def _calculate_dhdlam_multiPos(self, positions: (t.List[float] or float)) -> (np.array or float):
        return np.array(list(map(self._calculate_dhdlam_singlePos, positions)))

    @classmethod
    def _check_positions_type_singlePos(cls, position: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param position: here the positions that shall be evaluated by the potential should be given.
        :type position:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """
        if (isinstance(position, numbers.Number)):
            return np.array([np.float(position) for x in range(cls.nStates)])
        elif(isinstance(position, Sized)):
            if(len(position) == cls.nStates and all([isinstance(p, numbers.Number) for p in position])):
                return position
            elif(isinstance(position, Sized) and len(position) == cls.nDim and isinstance(position[0], numbers.Number)):
                return np.array([np.float(position[0]) for x in range(cls.nStates)])
        else:
            raise Exception("Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))

    @classmethod
    def _check_positions_type_multiPos(cls,
                                        positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:  t.Union[t.Iterable[numbers.Number], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """
        if (isinstance(positions, numbers.Number)):  # single number
            return np.array([positions for p in range(cls.nStates)], ndmin=1)
        elif (isinstance(positions, Iterable)):
            if (len(positions) != cls.nStates and all([isinstance(x, numbers.Number) for x in positions])):  # list with numbers
                return np.array(positions, ndmin=1)
            elif (len(positions) != cls.nStates and all([isinstance(x, numbers.Number) for x in positions])):  # list with numbers
                return np.array([positions  for p in range(cls.nStates)], ndmin=1)
            else:
                raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=2 potential Dimensions 1")
        else:
            if (type(positions) == type(None)):
                raise Exception("potential got None as position")
            else:
                raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=" + str(
                    len(positions)) + " potential Dimensions " + str(cls.nDim))

    def _calculate_energies_singlePos(self, position: float)  -> float:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos_singlePos(self, position: float)  -> float:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

class linCoupledHosc(_perturbedPotential1DCls):
    nStates = 2

    def __init__(self, ha:_potential1DCls=harmonicOsc(fc=1.0, x_shift=0.0), hb:_potential1DCls=harmonicOsc(fc=11.0, x_shift=0.0), lam:float=0):
        super().__init__()

        self.ha = ha
        self.hb = hb
        self.ha._set_singlePos_mode()
        self.hb._set_singlePos_mode()
        self.lam = lam
        self.couple_H = lambda Va, Vb: (1.0 - self.lam) * Va + self.lam * Vb

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.couple_H(float(self.ha.ene(position[0])), float(self.hb.ene(position[1])))

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return self.couple_H(float(self.ha.dhdpos(position[0])), float(self.hb.dhdpos(position[1])))

    def _calculate_dhdlam_singlePos(self, position:float) -> (np.array or float):
        return float(self.hb.ene(position[0])) - float(self.ha.ene(position[1]))


class expCoupledHosc(_perturbedPotential1DCls):
    nStates = 2

    def __init__(self, ha=harmonicOsc(fc=1.0, x_shift=0.0), hb=harmonicOsc(fc=11.0, x_shift=0.0), s=1.0, temp=300.0,
                 lam: float = 0.0):
        super().__init__()

        self.ha = ha
        self.hb = hb
        self.s = s
        self.beta = const.gas_constant / 1000.0 * temp
        self.lam = lam

        self.couple_H = lambda Va, Vb: -1.0 / (self.beta * self.s) * np.log(
            self.lam * np.exp(-self.beta * self.s * Vb) + (1.0 - self.lam) * np.exp(-self.beta * self.s * Va))

        self.couple_H_dhdpos = lambda Va, Vb: self.lam * Vb * np.exp(-self.beta * self.s * Vb) + (1.0 - self.lam) * \
                                           Va * np.exp(-self.beta * self.s * Va)

        self.couple_H_dhdlam = lambda Va,Vb: -1.0 / (self.beta * self.s) * 1.0 / (
                    self.lam * np.exp(-self.beta * self.s * Vb) + (1.0 - self.lam) * np.exp(
                -self.beta * self.s * Va)) * (np.exp(-self.beta * self.s * Vb) - np.exp(
            -self.beta * self.s * Va))

    def _calculate_energies_singlePos(self, position: float) -> float:
        return self.couple_H(float(self.ha.ene(position)), float(self.hb.ene(position)))

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return self.couple_H_dhdpos(float(self.ha.dhdpos(position)), float(self.hb.dhdpos(position)))

    def _calculate_dhdlam_singlePos(self, position:float) -> (np.array or float):
        return self.couple_H_dhdlam(float(self.ha.ene(position)), float(self.hb.ene(position)))


class pertHarmonicOsc(_perturbedPotential1DCls):
    """
        .. autoclass:: pertHarmonixsOsc1D
    """
    name = "perturbed Harmonic Oscilator"
    nStates = 2

    def __init__(self, fc=1.0, alpha=10.0, gamma=0.0, lam: float = 0.0):
        '''
        Initializes a potential of the form V = 0.5 * (1 + alpha * lam) * fc * (pos - gamma * lam) ** 2
        :param fc: force constant
        :param alpha: perturbation parameter for width of harmonic oscillator
        :param gamma: perturbation parameter for position of harmonic oscillator
        #TODO: reshap this implementation for using harmonic pot 1D
        '''
        super().__init__()

        self.fc = fc
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def _calculate_energies_singlePos(self, position: float) -> float:
        return 0.5 * (1.0 + self.alpha * self.lam) * self.fc * (position - self.gamma * self.lam) ** 2

    def _calculate_dhdpos_singlePos(self, position: float) -> float:
        return (1.0 + self.alpha * self.lam) * self.fc * (position - self.gamma * self.lam)

    def _calculate_dhdlam_singlePos(self, Va_t:float, Vb_t:float) -> (np.array or float):
        return False


class envelopedPotential(ND.envelopedPotential):
    """
    .. autoclass:: envelopedPotential
    """
    V_is:t.List[_potential1DCls] = None
    E_is:t.List[float] = None
    nStates:int = None
    s:float = None
    nDim:int = 1

    def __init__(self, V_is: t.List[_potential1DCls], s: float = 1.0, Eoff_i: t.List[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        super().__init__(V_is=V_is, s=s, Eoff_i=Eoff_i)
    # each state gets a position list

    def _check_positions_type(self, positions: (numbers.Number or t.Iterable[float] or t.Iterable[t.Iterable[float]])) -> np.array:
        if (isinstance(positions, numbers.Number)):
            return np.array([positions for state in range(self.nStates)], ndmin=1)
        elif(isinstance(positions, Iterable) and all([isinstance(position, numbers.Number) for position in positions])):
            return np.array([positions for state in range(self.nStates)], ndmin=1)
        elif (isinstance(positions, Iterable) and len(positions) == len(self.V_is) and all([isinstance(pos, Iterable) for pos in positions])):
            return np.array([positions for state in range(self.nStates)], ndmin=2)
        else:
            raise Exception("This is an unknown type of Data structure: " + str(type(positions)) + "\n" + str(positions))

    def _calculate_dhdpos_singlePos(self, position:(t.Iterable[float])) -> np.array:
        """
        :warning : Implementation is not entirly correct!
        :param positions:
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

        #prefactors = np.array([np.zeros(len(positions[0])) for x in range(len(positions))])
        #todo: error this should be ref pot fun not sum of all pots
        #FIX FROM HERE ON

        prefactors = np.array([list(map(lambda pos, posSum: list(map(lambda dimPos, dimPosSum: 1 - np.divide(dimPos, dimPosSum), pos, posSum)), Vn_ene, V_Is_posDim_eneSum)) for Vn_ene in V_Is_ene])
        ##print("preFactors: ",prefactors.shape, "\n\t", prefactors,  "\n\t", prefactors.T)
        dhdpos_state_scaled = np.multiply(prefactors, V_Is_dhdpos)
        #print("dhdpos_scaled", dhdpos_state_scaled.shape, "\n\t", dhdpos_state_scaled, "\n\t", dhdpos_state_scaled.T    )

        #dhdpos_R = [  for dhdpos_state in dhdpos_state_scaled]

        dhdpos_R = []
        for position in range(len(position[0])):
            dhdposR_position = []
            for dimPos in range(len(position[0][0])):
                dhdposR_positionDim = 0
                for state in range(len(V_Is_ene)):
                    dhdposR_positionDim = np.add(dhdposR_positionDim, dhdpos_state_scaled[state, position, dimPos])
                dlist = [dhdposR_positionDim]
                dlist.extend(dhdpos_state_scaled[:, position, dimPos])
                dhdposR_position.append(dlist)
            dhdpos_R.append(dhdposR_position)

        return np.array(dhdpos_R)

    def _calculate_energies_singlePos(self, position:(t.Iterable[float]))  -> np.array:
        #print(position)
        partA = np.multiply(-self.s, np.subtract(self.V_is[0].ene(position[0]), self.Eoff_i[0]))
        partB = np.multiply(-self.s, np.subtract(self.V_is[1].ene(position[1]), self.Eoff_i[1]))
        #print("OH", self.V_is[1].ene(position))

        #print("partA", partA)
        #print("partB", partB)
        sum_prefactors = np.add(max(partA, partB), np.log(np.add(1, np.exp(np.subtract(min(partA, partB), max(partA, partB))))))

        # more than two states!
        for state in range(2, self.nStates):
            partN = np.multiply(-self.s, np.subtract(self.V_is[state].ene(position[state]), self.Eoff_i[state]))
            sum_prefactors = np.add(np.max([sum_prefactors, partN]), np.log(np.add(1, np.exp(np.subtract(np.min([sum_prefactors, partN]), np.max([sum_prefactors, partN]))))))

        #print(sum_prefactors)
        Vr = np.multiply(np.divide(-1, float(self.s)), sum_prefactors)
        return Vr


class envelopedDoubleWellPotential(ND.envelopedPotential):
    def __init__(self, y_shifts: list = None, x_shifts=None,
                 smoothing: float = 1, fcs=None):
        if (y_shifts == None):
            y_shifts = [0, 0]
        if (x_shifts == None):
            x_shifts = [-1, 1]
        if (fcs == None):
            fcs = [1, 1]

        V_is = [harmonicOsc(x_shift=x_shift, y_shift=y_shift, fc=fc)
                for y_shift, x_shift, fc in zip(y_shifts, x_shifts, fcs)]
        super().__init__(V_is=V_is, s=smoothing)


if __name__ == "__main__":
    pass


"""
Module: Potential
This module shall be used to implement subclasses of Potential. This module contains all available potentials.
"""

import math
import numpy as np
import typing as t
import numbers
import scipy.constants as const
from collections.abc import Iterable

from Ensembler.src.potentials import ND

class _potential1DCls(ND._potentialNDCls):
    '''
        .. autoclass:: _potentialCls

        This class is the
    '''

    nDim:int = 1

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

        if(isinstance(positions, numbers.Number)):
            return np.array(positions, ndmin=1)
        elif (isinstance(positions, Iterable) and all([isinstance(x, numbers.Number) for x in positions])):
            return np.array(positions, ndmin=1)
        else:
            raise Exception("list dimensionality does not fit to potential dimensionality! len(list)=" + str(
                len(positions)) + " potential Dimensions " + str(cls.nDim))

    def ene(self, positions:(t.Iterable[float] or np.array or float)) -> (t.List[float] or float):
        '''
        calculates energy of particle
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: energy
        '''

        positions = self._check_positions_type(positions)
        return self._calculate_energies(positions)

    def dhdpos(self, positions:(t.Iterable[float] or np.array  or float)) -> (t.List[float] or float):
        '''
        calculates derivative with respect to position
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdpos(positions)


"""
    SIMPLE POTENTIALS
"""

class flat_well1D(_potential1DCls):
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
        super(_potential1DCls, self).__init__()
        self.x_min = min(x_range)
        self.x_max = max(x_range)
        self.y_max = y_max
        self.y_min = y_min

    def _calculate_energies(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array([self.y_min if (pos >= self.x_min and pos <= self.x_max) else self.y_max for pos in positions])

    def _calculate_dhdpos(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.zeros(len(positions))

class harmonicOsc1D(_potential1DCls):
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
        super(harmonicOsc1D, self).__init__()
        self.fc = fc
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _calculate_energies(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array(list(map(lambda pos: 0.5 * self.fc * (pos - self.x_shift) ** 2 - self.y_shift, positions)))

    def _calculate_dhdpos(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array(list(map(lambda pos: self.fc * (pos - self.x_shift), positions)))

class wavePotential1D(_potential1DCls):
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
        super(_potential1DCls).__init__()
        self.amplitude = amplitude
        self.multiplicity = multiplicity
        self.y_offset = y_offset
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
            self._calculate_energies = lambda positions: np.array(list(
                map(lambda x: self.amplitude * math.cos(self.multiplicity * (x + self.phase_shift)) + self.y_offset, positions)))
            self._calculate_dhdpos = np.array(lambda positions: list(
                map(lambda x: self.amplitude * math.sin(self.multiplicity * (x + self.phase_shift)) + self.y_offset, positions)))
        else:
            self._calculate_energies = lambda positions: np.array(list(
                map(lambda x: self.amplitude * math.cos(self.multiplicity * (x + self.phase_shift)) + self.y_offset,
                    np.deg2rad(positions))))
            self._calculate_dhdpos = lambda positions: np.array(list(
                map(lambda x: self.amplitude * math.sin(self.multiplicity * (x + self.phase_shift)) + self.y_offset,
                    np.deg2rad(positions))))

class torsionPotential1D(_potential1DCls):
    '''
        .. autoclass:: Torsion Potential
    '''
    name:str = "Torsion Potential"

    phase:float = 1.0
    wave_potentials:t.Iterable[wavePotential1D] = []

    def __init__(self, wave_potentials: t.Union[t.Iterable[wavePotential1D], wavePotential1D]):
        '''
        initializes torsions Potential
        '''
        super(_potential1DCls).__init__()
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
        elif(isinstance(wave_potentials, wavePotential1D)):
            self.wave_potentials = wave_potentials
            self._calculate_energies = lambda positions: self.wave_potentials.ene(positions)
            self._calculate_dhdpos = lambda positions: self.wave_potentials.dhdpos(positions)
        else:
            raise IOError("the provided potential was not recognizable.")

class coulombPotential1D(_potential1DCls):
    name = "Coulomb Potential"

    epsilon:float

    coulombLaw = lambda t, q1, q2, r, epsilon: np.divide(np.multiply(q1, q2), np.multiply(r, epsilon * 4 * math.pi))
    dcoulombLawdr = lambda t, q1, q2, r, epsilon: np.divide(np.multiply(-q1, q2), np.multiply(np.square(r), epsilon * 4 * math.pi))

    def __init__(self, q1=1, q2=1, epsilon=1):
        self.q1 = q1
        self.q2 = q2
        self.epsilon = epsilon
        pass

    def _calculate_energies(self, distances: t.List[float]):
        coulombLaw_curry = lambda x: self.coulombLaw(self.q1, self.q2, x, self.epsilon)
        return np.array(list(map(coulombLaw_curry, distances)))

    def _calculate_dhdpos(self, distances: t.List[float]):
        dcoulombLaw_currydr= lambda x: self.dcoulombLawdr(self.q1, self.q2, x, self.epsilon)
        return np.array(list(map(dcoulombLaw_currydr, distances)))

class lennardJonesPotential1D(_potential1DCls):
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
        x_shift = x_shift
        y_shift = y_shift

    def _calculate_energies(self, positions: (t.Iterable[float] or np.array or float)) -> np.array:
        return np.array([np.subtract(np.divide(self.c12, np.power(pos, 12)), np.divide(self.c6, np.power(pos, 6))) for pos in positions])

    def _calculate_dhdpos(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array([6 * ((2 * self.c12) - (pos ** 6 * self.c6)) / pos ** 13 for pos in positions])


class doubleWellPot1D(_potential1DCls):
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
        super(_potential1DCls).__init__()
        self.Vmax = Vmax
        self.a = a
        self.b = b

    def _calculate_energies(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array([*map(lambda pos: self.Vmax / self.b ** 4 * ((pos - self.a / 2) ** 2 - self.b ** 2) ** 2, positions)])

    def _calculate_dhdpos(self, positions: (t.Iterable[float] or np.array or float)) ->  np.array:
        return np.array([*map(lambda pos: 4 * self.Vmax / self.b ** 4 * ((pos - self.a / 2) ** 2 - self.b ** 2) * (pos - self.a / 2), positions)])


"""
    PERTURBED POTENTIALS
"""


class _perturbedPotential1DCls(_potential1DCls):
    """
        .. autoclass:: perturbedPotentialCls
    """
    lam:float

    def __init__(self, lam: float = 0.0):
        '''
        Initializes a potential of the form V = 0.5 * (1 + alpha * lam) * fc * (pos - gamma * lam) ** 2
        :param fc: force constant
        :param alpha: perturbation parameter for width of harmonic oscillator
        :param gamma: perturbation parameter for position of harmonic oscillator
        '''
        super(_potential1DCls).__init__()
        self.lam = lam

    def _calculate_energies(self, positions: t.List[float], lam: float = 1.0):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos(self, positions: t.List[float], lam: float = 1.0):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdlam(self, positions: t.List[float], lam: float = 1.0):
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def set_lam(self, lam: float):
        self.lam = lam

    def ene(self, positions: (t.List[float] or float)) -> (t.List[float] or float):
        '''
        calculates energy of particle
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: energy
        '''
        positions = self._check_positions_type(positions)
        enes = self._calculate_energies(positions)
        return enes

    def dhdpos(self, positions: (t.List[float] or float)) -> (t.List[float] or float):
        '''
        calculates derivative with respect to position
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dpos
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdpos(positions)

    def dhdlam(self, positions: (t.List[float] or float)) -> (t.List[float] or float):
        '''
        calculates derivative with respect to lambda value
        :param lam: alchemical parameter lambda
        :param pos: position on 1D potential energy surface
        :return: derivative dh/dlan
        '''
        positions = self._check_positions_type(positions)
        return self._calculate_dhdlam(positions)


class linCoupledHosc(_perturbedPotential1DCls):
    def __init__(self, ha=harmonicOsc1D(fc=1.0, x_shift=0.0), hb=harmonicOsc1D(fc=11.0, x_shift=0.0), lam=0):
        super(_perturbedPotential1DCls).__init__()

        self.ha = ha
        self.hb = hb
        self.lam = lam
        self.couple_H = lambda Va, Vb: (1.0 - self.lam) * Va + self.lam * Vb

    def _calculate_energies(self, positions: (t.List[float] or float)) -> (np.array or float):
        return np.array(list(map(self.couple_H, self.ha.ene(positions), self.hb.ene(positions))))

    def _calculate_dhdpos(self, positions: (t.List[float] or float)) -> (np.array or float):
        return np.array(list(map(self.couple_H, self.ha.dhdpos(positions), self.hb.dhdpos(positions))))

    def _calculate_dhdlam(self, positions: (t.List[float] or float)) -> (np.array or float):
        return np.array(list(map(lambda Va_t, Vb_t: Vb_t - Va_t, self.ha.ene(positions), self.hb.ene(positions))))


class expCoupledHosc(_perturbedPotential1DCls):
    def __init__(self, ha=harmonicOsc1D(fc=1.0, x_shift=0.0), hb=harmonicOsc1D(fc=11.0, x_shift=0.0), s=1.0, temp=300.0,
                 lam: float = 0.0):
        super(_perturbedPotential1DCls).__init__()

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

    def _calculate_energies(self, positions: (t.List[float] or float)) -> (np.array or float):
        return np.array(list(map(self.couple_H, self.ha.ene(positions), self.hb.ene(positions))))

    def _calculate_dhdpos(self, positions: (t.List[float] or float), lam: float = 1.0) -> (np.array or float):
        return np.array(list(map(self.couple_H_dhdpos, self.ha.ene(positions), self.hb.ene(positions))))

    def _calculate_dhdlam(self, positions: (t.List[float] or float), lam: float = 1.0) -> (np.array or float):
        return np.array(list(map(self.couple_H_dhdlam, self.ha.ene(positions), self.hb.ene(positions))))


class pertHarmonicOsc1D(_perturbedPotential1DCls):
    """
        .. autoclass:: pertHarmonixsOsc1D
    """
    name = "perturbed Harmonic Oscilator"

    def __init__(self, fc=1.0, alpha=10.0, gamma=0.0, lam: float = 0.0):
        '''
        Initializes a potential of the form V = 0.5 * (1 + alpha * lam) * fc * (pos - gamma * lam) ** 2
        :param fc: force constant
        :param alpha: perturbation parameter for width of harmonic oscillator
        :param gamma: perturbation parameter for position of harmonic oscillator
        '''
        super(_perturbedPotential1DCls).__init__()

        self.fc = fc
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def _calculate_energies(self, positions: (t.List[float] or float)) -> np.array:
        return np.array([0.5 * (1.0 + self.alpha * self.lam) * self.fc * (pos - self.gamma * self.lam) ** 2 for pos in positions])

    def _calculate_dhdpos(self, positions: (t.List[float] or float)) -> np.array :
        return np.array([(1.0 + self.alpha * self.lam) * self.fc * (pos - self.gamma * self.lam) for pos in positions])

    def _calculate_dhdlam(self, positions: (t.List[float] or float)) -> np.array :
        return np.array([0.5 * self.alpha * self.fc * (pos - self.gamma * self.lam) ** 2 - (
                1.0 + self.alpha * self.lam) * self.fc * self.gamma * (pos - self.gamma * self.lam) for pos in
                positions])


class envelopedDoubleWellPotential1D(ND.envelopedPotential):
    def __init__(self, y_shifts: list = None, x_shifts=None,
                 smoothing: float = 1, fcs=None):
        if (y_shifts == None):
            y_shifts = [0, 0]
        if (x_shifts == None):
            x_shifts = [-1, 1]
        if (fcs == None):
            fcs = [1, 1]

        V_is = [harmonicOsc1D(x_shift=x_shift, y_shift=y_shift, fc=fc)
                for y_shift, x_shift, fc in zip(y_shifts, x_shifts, fcs)]
        super().__init__(V_is=V_is, s=smoothing)


if __name__ == "__main__":
    pass


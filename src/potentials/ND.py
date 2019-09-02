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
    @nullState
    @Strategy Pattern
    '''
    name:str = "Unknown"
    nDim:int = -1
    nStates:int = 1
    _no_Type_check:bool=False
    _singlePos_mode:bool = False


    def __init__(self, nDim:int=-1):
        self.nDim = nDim
        self._calculate_energies=self._calculate_energies_multiPos
        self._calculate_dhdpos=self._calculate_dhdpos_multiPos
        self._check_positions_type= self._check_positions_type_multiPos

    def __name__(self)->str:
        return str(self.name)

    """
        public
    """
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
        private
    """
    """
            dummies
    """
    @classmethod
    def _check_positions_type(cls, positions: t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]) -> np.array:
        """
            .. autofunction:: _check_positions_type
            This function is parsing and checking all possible inputs to avoid misuse of the functions.
        :param positions: here the positions that shall be evaluated by the potential should be given.
        :type positions:   t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]
        :return: returns the evaluated potential values
        :return type: t.List[float]
        """
        raise Exception(__name__+"_Dummy Was not initialized! please call super constructor "+__class__.__name__+"!")

    def _calculate_energies(cls, positions: t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]) -> np.array:
        """
            .. autofunction:: _calculate_energies

        :param positions:
        :type positions:  t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]
        :return:
        :return type: np.array
        """
        raise Exception(__name__+"_Dummy Was not initialized! please call super constructor "+__class__.__name__+"!")

    def _calculate_dhdpos(cls, positions: t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]) -> np.array:
        """
            .. autofunction:: _calculate_dhdpos

        :param positions:
        :type positions:  t.Union[t.Iterable[numbers.Number], t.Iterable[t.Iterable[numbers.Number]], numbers.Number]
        :return:
        :return type: np.array
        """
        raise Exception(__name__ + "_Dummy Was not initialized! please call super constructor " + __class__.__name__ + "!")


    """
            type Juggeling and interface methods setting 
    """
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
        #array
        if(isinstance(position, numbers.Number)):
            return position
        elif(isinstance(position, Iterable)):
            if(cls.nDim == 1 and isinstance(position, numbers.Number)):
                return position[0]
            elif((len(position) == cls.nDim or cls.nDim == -1) and all([isinstance(position, numbers.Number) for position in position])):   #position[dimPos]
                return np.array(position, ndmin=1)
            else:
                raise Exception("Input Type dimensionality does not fit to potential dimensionality! Input: " + str(position))
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
        # array
        if(isinstance(positions, Iterable)):
            if(all([isinstance(position, Iterable) and (len(position) == cls.nDim or cls.nDim == 0) for position in positions])):   #positions[[position[dimPos]]
                if(all([all([isinstance(dimPos, numbers.Number) for dimPos in position]) for position in positions])):
                    return np.array(positions, ndmin=2)
                else:
                    raise Exception()
            elif (all([isinstance(position, numbers.Number) for position in positions])):  # positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            elif (all([isinstance(position, Iterable) and all([isinstance(x, numbers.Number) for x in position]) for position in positions])):  # positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            elif((cls.nDim == 1 or cls.nDim ==0) and all([isinstance(position, numbers.Number) for position in positions])):   #positions[[position[dimPos]]
                return np.array(positions, ndmin=2)
            else:
                raise Exception("Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))
        elif ((cls.nDim ==1 or cls.nDim ==-1) and isinstance(positions, numbers.Number)):
            return np.array(positions, ndmin=2)
        else:
            raise Exception("Input Type dimensionality does not fit to potential dimensionality! Input: " + str(positions))

    def _calculate_dhdpos_singlePos(self, positions:(t.Iterable[float])) -> np.array:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_energies_singlePos(self, position:(t.Iterable[float]))  -> np.array:
        raise NotImplementedError("Function " + __name__ + " was not implemented for class " + str(__class__) + "")

    def _calculate_dhdpos_multiPos(self, positions: (t.Iterable[float] or t.Iterable[t.Iterable[float]] or np.array)) ->  np.array:
        return np.array(list(map(self._calculate_dhdpos_singlePos, positions)))

    def _calculate_energies_multiPos(self, positions: (t.Iterable[float] or np.array)) ->  np.array:
        """
        ..autofunction :: _calculate_energies_multiPos

        :return:  -
        """
        return np.array(list(map(self._calculate_energies_singlePos, positions)))

    """
            Input - Options
                For Performance or easier use!
    """
    def _set_singlePos_mode(self):
        """
        ..autofunction :: _set_singlePos_mode

        :return:  -
        """
        self._singlePos_mode = True
        self._check_positions_type = self._check_positions_type_singlePos
        self._calculate_energies = self._calculate_energies_singlePos
        self._calculate_dhdpos = self._calculate_dhdpos_singlePos
        print(__name__+"in _set_singlePos_mode ",self.nDim)

    def _set_multiPos_mode(self):
        """
        ..autofunction :: _set_multiPos_mode

        :return:  -
        """
        self._singlePos_mode = False
        self._check_positions_type = self._check_positions_type_multiPos
        self._calculate_energies = self._calculate_energies_multiPos
        self._calculate_dhdpos = self._calculate_dhdpos_multiPos

    def _set_no_type_check(self):
        """
        ..autofunction :: _set_no_type_check
            This function is trying to speed up execution for cases, in that the position Type is known to be correct (system integration) ...

        :return:  -
        """
        _no_Type_check = True
        self._check_positions_type = lambda x: x


    def _set_type_check(self):
        """
        ..autofunction :: _set_type_check
            This function is setting the default potential Value, to allow secure execution in small code snippets
        :return:  -
        """
        _no_Type_check = False
        if(self._singlePos_mode):
            self._check_positions_type = self._check_positions_type_singlePos
        else:
            self._check_positions_type = self._check_positions_type_multiPos

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

    def _calculate_dhdpos_singlePos(self, position:(t.Iterable[float])) -> np.array:
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
    V_is:t.List[_potentialNDCls] = None
    E_is:t.List[float] = None
    nStates:int = None
    s:float = None
    nStates:int = 0

    def __init__(self, V_is: t.List[_potentialNDCls], s: float = 1.0, Eoff_i: t.List[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        """

        import numpy as np
        from Ensembler.src import potentials as pot
        shift=90
        positions = np.array([(x_t, y_t) for x_t in range(10) for y_t in range(10)])
        V1 = pot.TwoD.wavePotential(phase_shift=(shift, shift), multiplicity=(3.0, 3.0), amplitude=(50.0, 50.0))
        V2 = pot.TwoD.wavePotential(phase_shift=(shift, shift), multiplicity=(3.0, 3.0), amplitude=(50.0, 50.0))
        edsPot = pot.ND.envelopedPotential(V_is=[V1, V2], s=1.0, Eoff_i=[-2, 0])
        edsPot.ene(positions)
        
        """

        # Todo: think about n states with each m dims.
        if(self.nDim == -1):
            self.nDim = V_is[0].nDim

        super().__init__(nDim=self.nDim)

        #check State number
        self.nStates = len(V_is)
        if (self.nStates < 2):
            raise IOError("It does not make sense enveloping less than two potentials!")
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

    def _check_positions_type_singlePos(cls, position: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        if (isinstance(position, numbers.Number)):
                return np.array([[[position]] for state in range(cls.nStates)], ndmin=3)
        elif (isinstance(position, Iterable)):
            if(all([isinstance(x, numbers.Number) for x in position])):    #ndim pot list
                return np.array([[position] for state in range(cls.nStates)], ndmin=2)
            elif(all([isinstance(x, Iterable) and all([isinstance(y, numbers.Number) for y in x]) for x in position])):    #nDim pos lis
                return np.array([position for position in range(cls.nStates)], ndmin=3)
            elif(all([isinstance(x, Iterable) and all([isinstance(y, Iterable) and all([isinstance(z, numbers.Number) for z in y] ) for y in x]) for x in position])):
                return np.array(position, ndmin=3)
            else:
                raise Exception("This is an unknown type of Data structure: " + str(type(position)) + "\n" + str(position))
        else:
            raise Exception("This is an unknown type of Data structure: " + str(type(position)) + "\n" + str(position))

    def _check_positions_type_multiPos(cls, positions: t.Union[t.Iterable[numbers.Number], numbers.Number]) -> np.array:
        if (isinstance(positions, numbers.Number)):
                return np.array([[[positions]] for state in range(cls.nStates)], ndmin=3)
        elif (isinstance(positions, Iterable)):
            if(all([isinstance(x, numbers.Number) for x in positions])):    #ndim pot list
                return np.array([[[p] for state in range(cls.nStates)] for p in positions], ndmin=3)
            elif(all([isinstance(x, Iterable) and all([isinstance(y, numbers.Number) for y in x]) for x in positions])):    #nDim pos lis
                return np.array([positions for position in range(cls.nStates)])
            elif(all([isinstance(x, Iterable) and all([isinstance(y, Iterable) and all([isinstance(z, numbers.Number) for z in y] ) for y in x]) for x in positions])):
                return np.array(positions, ndmin=3)
            else:
                raise Exception("This is an unknown type of Data structure, wrapped by a Iterable: " + str(type(positions)) + "\n" + str(positions))
        else:
            raise Exception("This is an unknown type of Data structure: " + str(type(positions)) + "\n" + str(positions))

    def _calculate_energies_singlePos(self, position:(t.Iterable[float])) -> np.array:
        #print("NDSi", position)

        partA = np.multiply(-self.s, np.subtract(self.V_is[0].ene(position[0]), self.Eoff_i[0]))
        partB = np.multiply(-self.s, np.subtract(self.V_is[1].ene(position[1]), self.Eoff_i[1]))
        #print("partA", partA)
        #print("partB", partB)

        sum_prefactors = max(partA, partB) + np.log(1 + np.exp(min(partA, partB) - max(partA, partB)))

        # more than two states!
        for state in range(2, self.nStates):
            partN = np.multiply(-self.s, np.subtract(self.V_is[state].ene(position[state]), self.Eoff_i[state]))
            sum_prefactors = max(sum_prefactors, partN) + np.log(1 + np.exp(min(sum_prefactors, partN) - max(sum_prefactors, partN)))

        #print(sum_prefactors)
        Vr = float(np.sum(np.multiply(np.divide(-1, float(self.s)), sum_prefactors)))
        return Vr

    def _calculate_dhdpos_singlePos(self, position:(t.Iterable[float])) -> np.array:
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

        print(position)
        prefactors = np.array([list(map(lambda dimPos: 1 - np.divide(dimPos, V_Is_posDim_eneSum), list(Vn_ene))) for Vn_ene in V_Is_ene])
        ##print("preFactors: ",prefactors.shape, "\n\t", prefactors,  "\n\t", prefactors.T)
        dhdpos_state_scaled = np.multiply(prefactors, V_Is_dhdpos)
        #print("dhdpos_scaled", dhdpos_state_scaled.shape, "\n\t", dhdpos_state_scaled, "\n\t", dhdpos_state_scaled.T    )

        #dhdpos_R = [  for dhdpos_state in dhdpos_state_scaled]

        dhdpos_R = []
        print("Ndim: ", self.nDim)
        for dimPos in range(self.nDim):
            dhdposR_positionDim = 0
            for state in range(len(V_Is_ene)):
                dhdposR_positionDim = np.add(dhdposR_positionDim, dhdpos_state_scaled[state, dimPos])
            dlist = [dhdposR_positionDim]
            dlist.extend(dhdpos_state_scaled[:, dimPos])
            dhdpos_R.append(dlist)

        return np.array(dhdpos_R)

    def _set_singlePos_mode(self):
        """
        ..autofunction :: _set_singlePos_mode

        :return:  -
        """
        self._singlePos_mode = True
        self._calculate_energies = self._calculate_energies_singlePos
        self._calculate_dhdpos = self._calculate_dhdpos_singlePos
        [V._set_singlePos_mode() for V in self.V_is]

    def _set_multiPos_mode(self):
        """
        ..autofunction :: _set_multiPos_mode

        :return:  -
        """
        super()._set_multiPos_mode()
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
    V_is:t.List[_potentialNDCls] = None
    E_is:t.List[float] = None
    nStates:int = None
    s:t.List[float] = None

    def __init__(self, V_is: t.List[_potentialNDCls], s: t.List[float], Eoff_i: t.List[float] = None):
        """

        :param V_is:
        :param s:
        :param Eoff_i:
        """
        super().__init__(V_is=V_is, Eoff_i=Eoff_i)
        self.s = s

    def _calculate_energiesND(self, positions: (t.List[float] or float)) -> np.array:
        partA = [-self.s[0] * (Vit - self.Eoff_i[0]) for Vit in self.V_is[0].ene(positions[0])]
        partB = [-self.s[1] * (Vit - self.Eoff_i[1]) for Vit in self.V_is[1].ene(positions[1])]
        sum_prefactors = np.array([list(map(lambda A_t, B_t: max(A_t, B_t) + np.log(1 + np.exp(min(A_t, B_t) - max(A_t, B_t))), A, B)) for A, B in
                          zip(partA, partB)])

        # more than two states!
        for state in range(2, self.nStates):
            partN = [np.multiply(-self.s[state], np.subtract(Vit, self.Eoff_i[state]))for Vit in self.V_is[state].ene(positions[state])]
            sum_prefactors = np.array(
                [list(map(lambda A_t, B_t: max(A_t, B_t) + math.log(1 + math.exp(min(A_t, B_t) - max(A_t, B_t))), A, B))
                 for A, B in
                 zip(sum_prefactors, partN)])

        Vr = [-1  * partitionF for partitionF in sum_prefactors]
        return np.array(Vr)

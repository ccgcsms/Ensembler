import os,sys
import unittest
import numpy as np
import numbers
from collections.abc import Iterable
sys.path.append(os.path.dirname(__file__+"/../.."))

from Ensembler.src.potentials import ND as pot

"""
TEST for Potentials 1D
"""
class potentialNDCls(unittest.TestCase):

    """
    TEST for Potential inputs
    """
    def test_check_positions_float_type(self):
        #check single Float
        position = 1.0
        expected =[[1.0]]
        checked_pos = pot._potentialNDCls._check_positions_type(positions=position)

        print(checked_pos)
        if(not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))

    def test_check_positions_npArray_type(self):
        #check nparray
        position = np.arange(1,10)
        expected = [[pos] for pos in position]
        checked_pos = pot._potentialNDCls._check_positions_type(positions=position)

        if(not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))


    def test_check_positions_list_type(self):
        #check LIST[Float]
        position = [1.0, 2.0, 3.0]
        expected = [[pos] for pos in position]

        checked_pos = pot._potentialNDCls._check_positions_type(positions=position)
        print(checked_pos)

        if(not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))

    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]
        expected = position
        checked_pos = pot._potentialNDCls._check_positions_type(positions=position)

        if(not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))

    def test_check_positions_2Dlist_type(self):
        position = [[1.0, 2.0], [3.0, 4.0]]
        expected = position
        checked_pos = pot._potentialNDCls._check_positions_type(positions=position)

        if(not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))
        elif(any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))

"""
TEST Potentials
"""
class potentialCls_flatwelND(unittest.TestCase):
    def test_constructor(self):
        pot.flat_wellND()

    def test_energies(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        expected_result = np.array([10, 0, 0, 0])

        potential = pot.flat_wellND(x_range=x_range, y_max=y_max, y_min=y_min)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdpos(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        potential = pot.flat_wellND(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([0, 0, 0, 0])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_energies2D(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [(-1, -1), (1, 0.5), (1, 10)]
        expected_result = np.array([20, 0, 10])

        potential = pot.flat_wellND(x_range=x_range, y_max=y_max, y_min=y_min)

        energies = potential.ene(positions)
        print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdpos2D(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [(0, 0), (1, 2), (1, 10)]
        potential = pot.flat_wellND(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([0, 0, 0])

        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

class potentialCls_harmonicOsc1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.harmonicOscND()

    def test_energies1D1Pos(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = 3
        expected_result = np.array([4.5])

        potential = pot.harmonicOscND(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_energiesND1Pos(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [0,1,2,0.5]
        expected_result = np.array([2.625])

        potential = pot.harmonicOscND(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdposND1Pos(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions = [0, 0.5, 1, 2]
        expected_result = np.array([3.5])

        potential = pot.harmonicOscND(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_energies2D(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [(0,1),(1,2),(3,4)]
        expected_result = np.array([0.5, 2.5, 12.5])

        potential = pot.harmonicOscND(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)
        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdpos2D(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions =[(0,1),(1,2),(3,4)]
        expected_result = np.array([1, 3, 7])

        potential = pot.harmonicOscND(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

class potentialCls_envelopedPotential(unittest.TestCase):
    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_envelopedPotentialMultiS(unittest.TestCase):
    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

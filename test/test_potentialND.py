import os,sys
import unittest
import numpy as np
import numbers
from collections.abc import Iterable
sys.path.append(os.path.dirname(__file__+"/../.."))

from numbers import Number
from Ensembler.src.potentials import ND as pot
from Ensembler.src.potentials import OneD as pot1

"""
TEST for Potentials ND
"""
class potentialNDCls(unittest.TestCase):

    """
    TEST for Potential inputs
    """
    def test_check_positions_float_type(self):
        #check single Float
        position = 1.0
        expected = position
        
        checked_pos = pot._potentialNDCls._check_positions_type_singlePos(position=position)

        print(checked_pos)
        if(not isinstance(checked_pos, Number)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: "+str(expected)+"\n"
                            "\tgot: "+str(checked_pos))

    def test_check_positions_npArray_type(self):
        #check nparray
        position = np.arange(1,10)
        expected = [[pos] for pos in position]
        checked_pos = pot._potentialNDCls._check_positions_type_multiPos(positions=position)

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
        expected = np.array(position, ndmin=2)

        checked_pos = pot._potentialNDCls._check_positions_type_multiPos(positions=position)

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
        expected = np.array(position, ndmin=2)

        checked_pos = pot._potentialNDCls._check_positions_type_multiPos(positions=position)

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
        expected = np.array(position, ndmin=2)
        checked_pos = pot._potentialNDCls._check_positions_type_multiPos(positions=position)

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
        pot.flat_well()

    def test_energies(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        expected_result = np.array([0, 10, 0, 0], ndmin=2)

        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies2D(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [(-1, -1), (1, 0.5), (1, 10)]
        expected_result = np.array([(10,10), (0,0), (0,10)])

        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)

        energies = potential.ene(positions)
        print("HA",energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([(0, 0, 0, 0)])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos2D(self):
        x_range = [0, 1]
        y_max = 10
        y_min = 0
        positions = [(0, 0), (1, 2), (1, 10)]
        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([(0,0), (0,0), (0,0)])

        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

class potentialCls_harmonicOscND(unittest.TestCase):
    def test_constructor(self):
        potential = pot.harmonicOsc()

    def test_energies1D1Pos(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = 3
        expected_result = np.array([4.5], ndmin=1)

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energiesND1Pos(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [0, 1, 2,0.5]
        expected_result = np.array(2.625, ndmin=1)

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdposND1Pos(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions = [0, 0.5, 1, 2]
        expected_result = np.array([0, 0.5, 1, 2], ndmin=2)

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies2D(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [(0,1),(1,2),(3,4)]
        expected_result = np.array([0.5, 2.5, 12.5])

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies),
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

    def test_dHdpos2D(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions =[(0,1),(1,2),(3,4)]
        expected_result = np.array([(0, 1), (1, 2), (3, 4)])

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

class potentialCls_envelopedPotential(unittest.TestCase):
    def test_constructor(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)

        potential = pot.envelopedPotential(V_is=[ha, hb])

    def test_check_positions_float(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = 4
        expected = np.array([[[4]], [[4]]])
        checked = potential._check_positions_type(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

    def test_check_positions_1DIterable(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = np.array([[4], [2], [3]])
        expected = np.array([[[4], [2], [3]], [[4], [2], [3]]], ndmin=3)
        checked = potential._check_positions_type(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

    def test_check_positions_2DIterable(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = np.array((4,1))
        expected = np.array([[[4],[4]],[[1],[1]]], ndmin=3)
        checked = potential._check_positions_type(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

    def test_check_positionsND1Pos(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = [0, 0.5, 1, 2]
        expected = np.array([[[0], [0]], [[0.5], [0.5]], [[1],[1]], [[2],[2]]], ndmin = 2)
        checked = potential._check_positions_type_multiPos(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

    def test_check_positions1DNPos(self):
        ha = pot1.harmonicOsc(x_shift=-5)
        hb = pot1.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = [0, 0.5, 1, 2]
        expected = np.array([[0, 0.5, 1, 2], [0, 0.5, 1, 2]], ndmin = 2)
        checked = potential._check_positions_type(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

        energies = potential.ene(positions)
        print("ENE> ", energies)


    def test_check_positionsNDPoscorrectType(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = [[[1, 2]], [[3, 4]]]
        expected = np.array([[[1, 2]], [[3, 4]]], ndmin=3)
        checked = potential._check_positions_type(positions=positions)

        print(checked)
        np.testing.assert_equal(checked, expected, "not the same sorry.")

    def test_energiesND1Pos(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)

        position = [5, -5, 0]
        expected_result = np.array([0., 0., 11.80685282], ndmin=2)

        potential = pot.envelopedPotential(V_is=[ha, hb])
        energies = potential.ene(position)

        print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

    def test_check_positions1DNPos(self):
        ha = pot1.harmonicOsc(x_shift=-5)
        hb = pot1.harmonicOsc(x_shift=5)
        potential = pot.envelopedPotential(V_is=[ha, hb])

        positions = [0, 0.5, 1, 2]
        expected = np.array([11.80685282, 10.11828465, 7.9999546, 4.5])
        energies = potential.ene(positions)

        np.testing.assert_almost_equal(energies, expected, err_msg="not the same sorry.", decimal=5)

    def test_dhdpos1DNPos(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        setattr(ha, "nDim", 1)
        setattr(hb, "nDim", 1)

        positions = [5, -5, 2, 0]
        expected_result = np.array([[-1.44827586,  1.0862069,  -2.53448276],
                                    [ 0.,          2.5,        -2.5       ]], ndmin=2)

        potential = pot.envelopedPotential(V_is=[ha, hb])
        potential._set_multiPos_mode()
        energies = potential.dhdpos(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

class potentialCls_envelopedPotentialMultiS(unittest.TestCase):
    def test_constructor(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        s = (1,1)

        potential = pot.envelopedPotentialMultiS(V_is=[ha, hb],s=s)

    def test_energies1D1Pos(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        s = (1,1)
        positions = [5, -5, 0]
        expected_result = np.array([[0., 0., 11.80685282]], ndmin=2)

        potential = pot.envelopedPotentialMultiS(V_is=[ha, hb],s=s)

        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

    def test_dhdpos1D(self):
        ha = pot.harmonicOsc(x_shift=-5)
        hb = pot.harmonicOsc(x_shift=5)
        s = (1,1)
        positions = [5, -5, 2, 0]
        expected_result = np.array([[[0., 0., 0., ],
                                     [0., 0., -0.],
                                     [-1.44827586, 1.0862069, -2.53448276],
                                     [0., 2.5, -2.5]]], ndmin=3)

        potential = pot.envelopedPotentialMultiS(V_is=[ha, hb],s=s)

        energies = potential.dhdpos(positions)
        print(str(energies))
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)


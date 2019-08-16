import os,sys
import unittest
import numpy as np
import numbers
from collections.abc import Iterable
sys.path.append(os.path.dirname(__file__+"/../.."))

from Ensembler.src.potentials import TwoD as pot

"""
TEST for Potentials ND
"""


class potential2DCls(unittest.TestCase):
    """
    TEST for Potential inputs
    """

    def test_check_position_1Dfloat_type(self):
        # check single Float
        position = 1.0
        expected = np.array(position, ndmin=2)

        try:
            checked_pos = pot._potential2DCls._check_positions_type(positions=position)
        except :
            print("got error")
            return 0

        print(checked_pos)
        print("Did not get an Error!")
        exit(1)

    def test_check_position_2Dfloat_type(self):
        # check single Float
        position = (1.0, 1.0)
        expected = np.array(position, ndmin=2)
        checked_pos = pot._potential2DCls._check_positions_type(positions=position)

        print(checked_pos)
        print("Did not get an Error!")
        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_2Dlist_type(self):
        # check LIST[Float]
        position = [(1.0, 2.0), (3.0, 4.0)]
        expected = np.array(position, ndmin=2)

        checked_pos = pot._potential2DCls._check_positions_type(positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, numbers.Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]

        try:
            checked_pos = pot._potential2DCls._check_positions_type(positions=position)
        except:
            print("got error")
            return 0

        print(checked_pos)
        print("Did not get an Error!")
        exit(1)

"""
Test Simple Potentials:
"""

class potentialCls_wavePotential2D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.wavePotential2D()

    def test_energies2D1Pos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = (0,0)
        expected_result = np.array([2])

        potential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies2DNPos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([2, 1, -1, -1, 2])

        potential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos2D1Pos(self):
        phase_shift= (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = (0,0)
        expected_result = np.array([0,0], ndmin=2)

        potential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos2DNPos(self):
        phase_shift = (0.0, 0.0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0.0)
        radians = False

        positions = [(0, 0), (90, 0), (180, 270), (90, 270), (270, 0), (360, 360)]
        expected_result = np.array([[0,0],[1,0],[0,-1],[1,-1],[-1,0],[0,0]])

        potential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                        y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies),
                                       err_msg="The results of " + potential.name + " are not correct!", decimal=8)

class potentialCls_torsionPotential(unittest.TestCase):
    def test_constructor_SinglePotential(self):
        WavePotential2 = pot.wavePotential2D()
        torsionPot = pot.torsionPotential2D(wave_potentials=[WavePotential2])

    def test_constructor_ListPotentials(self):
        WavePotential = pot.wavePotential2D()
        WavePotential2 = pot.wavePotential2D()
        potential = pot.torsionPotential2D(wave_potentials=[WavePotential, WavePotential2])

    def test_energies_singlepot(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([2, 1, -1, -1, 2])

        WavePotential2 = pot.wavePotential2D()
        torsionPot = pot.torsionPotential2D(wave_potentials=[WavePotential2])
        energies = torsionPot.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+torsionPot.name+" are not correct!", decimal=8)

    def test_energies_singlepo_list(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([2, 1, -1, -1, 2])

        WavePotential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential2D(wave_potentials=[WavePotential])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies(self):
        phase_shift = (0.0, 0)
        multiplicity = (1.0, 1.0)
        amplitude = (1.0, 1.0)
        y_offset = (0.0, 0, 0)
        radians = False

        positions = [(0,0), (90,0), (180,270), (270,180), (360,360)]
        expected_result = np.array([4, 2,  -2, -2, 4])

        WavePotential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential2D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_energies_phase_shifted(self):
        phase_shift1 = (0.0,0)
        phase_shift2 = (180,180)
        multiplicity = (1.0,1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0,0,0)
        radians = False

        positions = [(0,0), (90,90), (180,0), (270,0), (360,0)]
        expected_result = np.array([0, 0, 0, 0, 0])

        WavePotential = pot.wavePotential2D(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential2D(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude,
                                             y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential2D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos(self):
        pass
        """
        phase_shift = (0.0,0)
        multiplicity = (1.0,1.0)
        amplitude = (1.0,1.0)
        y_offset = (0.0,0,0)
        radians = False

        positions = [(0,0), (90,90), (180,0), (270,0), (360,0)]
        expected_result = np.array([[0], [0], [0], [0], [0]])

        WavePotential = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential2D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential2D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

        """
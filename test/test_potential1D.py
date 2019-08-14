import os,sys
import unittest
import numpy as np
import numbers
from collections.abc import Iterable
sys.path.append(os.path.dirname(__file__+"/../.."))

from Ensembler2.src.potentials import OneD as pot

"""
TEST for Potentials 1D
"""
class potential1DCls(unittest.TestCase):

    """
    TEST for Potential inputs
    """
    def test_check_positions_float_type(self):
        #check single Float
        position = 1.0
        checked_pos = pot._potential1DCls._check_positions_type(positions=position)

        print(checked_pos)
        if(not isinstance(checked_pos, Iterable)):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif(any([not isinstance(pos, numbers.Number) for pos in checked_pos])):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")

    def test_check_positions_npArray_type(self):
        #check nparray
        position = np.arange(1,10)
        checked_pos = pot._potential1DCls._check_positions_type(positions=position)
        print(checked_pos)

        if (not isinstance(checked_pos, Iterable)):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, numbers.Number) for pos in checked_pos])):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")


    def test_check_positions_list_type(self):
        #check LIST[Float]
        position = [1.0, 2.0, 3.0]
        checked_pos = pot._potential1DCls._check_positions_type(positions=position)
        print(checked_pos)

        if (not isinstance(checked_pos, Iterable)):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, numbers.Number) for pos in checked_pos])):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")


    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]
        expected = 'list dimensionality does not fit to potential dimensionality! len(list)=1 potential Dimensions 1'
        try:
            checked_pos = pot._potential1DCls._check_positions_type(positions=position)
        except Exception as err:
            print(err.args)
            self.assertEqual(expected, err.args[0])
            print("Found Err")
            return 0
        print("Did finish without error!")
        exit(1)

    def test_check_positions_2Dlist_type(self):
        position = [[1.0, 2.0], [3.0, 4.0]]
        expected = 'list dimensionality does not fit to potential dimensionality! len(list)=2 potential Dimensions 1'
        try:
            checked_pos = pot._potential1DCls._check_positions_type(positions=position)
        except Exception as err:
            print(err.args)
            self.assertEqual(expected, err.args[0])
            print("Found Err")
            return 0
        print("Did finish without error!")
        exit(1)

"""
TEST for Potentials 1D
"""
class potentialCls_flatwell(unittest.TestCase):
    def test_constructor(self):
        pot.flat_well1D()

    def test_energies(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        expected_result = np.array([0, 10, 0, 0])

        potential = pot.flat_well1D(x_range=x_range, y_max=y_max, y_min=y_min)

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        print(energies)
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdpos(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        potential = pot.flat_well1D(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([0, 0, 0, 0])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        print(energies)
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

class potentialCls_harmonicOsc1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.harmonicOsc1D()

    def test_energies(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [0,2,1,0.5]
        expected_result = np.array([0, 2, 0.5, 0.125])

        potential = pot.harmonicOsc1D(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

    def test_dHdpos(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions = [0,0.5, 1, 2]
        expected_result = np.array([0, 0.5, 1, 2])

        potential = pot.harmonicOsc1D(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertEqual(first=Vexp, second=Vcalc, msg="The results were not correct!")

class potentialCls_wavePotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.wavePotential1D()

    def test_energies(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([1, 0,  -1, 0, 1])

        potential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_dHdpos(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([0, 1,  0, -1, 0])

        potential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

class potentialCls_torsionPotential(unittest.TestCase):
    def test_constructor_SinglePotential(self):
        WavePotential = pot.wavePotential1D()
        potential = pot.torsionPotential1D(wave_potentials=WavePotential)

    def test_constructor_ListPotentials(self):
        WavePotential = pot.wavePotential1D()
        WavePotential2 = pot.wavePotential1D()
        potential = pot.torsionPotential1D(wave_potentials=[WavePotential, WavePotential2])

    def test_energies_singlepot(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential1D(wave_potentials=WavePotential)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_energies_singlepo_listt(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential1D(wave_potentials=[WavePotential])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_energies(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([2, 0,  -2, 0, 2])

        WavePotential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential1D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
           self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_energies_phase_shifted(self):
        phase_shift1 = 0.0
        phase_shift2 = 180
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([0, 0, 0, 0, 0])

        WavePotential = pot.wavePotential1D(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude,
                                            y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential1D(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude,
                                             y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential1D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_dHdpos(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([0, 2,  0, -2, 0])

        WavePotential = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential1D(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential1D(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
           self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

class potentialCls_coulombPotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.coulombPotential1D()

    def test_energies(self):
        q1 = 1
        q2 = 1
        epsilon = 1

        positions = [0, 0.2, 0.5, 1, 2, 360]
        expected_result = np.array([np.inf, 0.3978, 0.15915,0.0795, 0.03978, 0.000221])

        potential = pot.coulombPotential1D(q1=q1, q2=q2, epsilon=epsilon)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_lennardJonesPotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.lennardJonesPotential1D()

    def test_energies(self):
        c6: float = 1**(-1)
        c12: float = 1**(-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3 ,6 ]
        expected_result = np.array([np.inf, 1.1990*10**14, 1.464*10**10, -1, 0])

        potential = pot.lennardJonesPotential1D(c6=c6, c12=c12, x_shift=x_shift, y_shift=y_shift)
        energies = potential.dhdpos(positions)

        from matplotlib import pyplot as plt
        plt.plot(energies)
        plt.ylim(-10,10)
        plt.show()

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        for Vexp, Vcalc in zip(expected_result, energies):
            self.assertAlmostEqual(first=Vexp, second=Vcalc, places=3, msg="The results were not correct!")

    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_doubleWellPot1D(unittest.TestCase):

    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_perturbedLinCoupledHosc(unittest.TestCase):
    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_perturbedExpCoupledHosc(unittest.TestCase):

    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

class potentialCls_perturbedHarmonicOsc1D(unittest.TestCase):
    def test_energies(self):
        raise NotImplementedError("Implement this test maaaaan!")
    def test_dHdpos(self):
        raise NotImplementedError("Implement this test maaaaan!")

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


if __name__ == '__main__':
    unittest.main()
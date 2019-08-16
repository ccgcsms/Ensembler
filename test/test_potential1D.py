import os,sys
import unittest
import numpy as np
import numbers
from collections.abc import Iterable
sys.path.append(os.path.dirname(__file__+"/../.."))

from Ensembler.src.potentials import OneD as pot


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
        expected = [1.0, 2.0, 3.0]
        checked_pos = pot._potential1DCls._check_positions_type(positions=position)

        if (not isinstance(checked_pos, Iterable)):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, numbers.Number) for pos in checked_pos])):
            print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")

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
        pot.flat_well()

    def test_energies(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        expected_result = np.array([0, 10, 0, 0])

        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)

        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of "+potential.name+" are not correct!")


    def test_dHdpos(self):
        x_range = [0,1]
        y_max = 10
        y_min = 0
        positions = [0,2,1,0.5]
        potential = pot.flat_well(x_range=x_range, y_max=y_max, y_min=y_min)
        expected_result = np.array([0, 0, 0, 0])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of "+potential.name+" are not correct!")

class potentialCls_harmonicOsc1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.harmonicOsc()

    def test_energies(self):
        fc= 1.0
        x_shift = 0.0
        y_shift = 0.0
        positions = [0,2,1,0.5]
        expected_result = np.array([0, 2, 0.5, 0.125])

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of "+potential.name+" are not correct!")


    def test_dHdpos(self):
        fc: float = 1.0
        x_shift:float = 0.0
        y_shift:float = 0.0
        positions = [0,0.5, 1, 2]
        expected_result = np.array([0, 0.5, 1, 2])

        potential = pot.harmonicOsc(fc=fc, x_shift=x_shift, y_shift=y_shift)

        energies = potential.dhdpos(positions)
        print(energies)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        self.assertListEqual(list(expected_result), list(energies), msg="The results of "+potential.name+" are not correct!")

class potentialCls_wavePotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.wavePotential()

    def test_energies(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([1, 0,  -1, 0, 1])

        potential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

    def test_dHdpos(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([0, 1,  0, -1, 0])

        potential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

class potentialCls_torsionPotential(unittest.TestCase):
    def test_constructor_SinglePotential(self):
        WavePotential = pot.wavePotential()
        potential = pot.torsionPotential(wave_potentials=WavePotential)

    def test_constructor_ListPotentials(self):
        WavePotential = pot.wavePotential()
        WavePotential2 = pot.wavePotential()
        potential = pot.torsionPotential(wave_potentials=[WavePotential, WavePotential2])

    def test_energies_singlepot(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                          y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential(wave_potentials=WavePotential)
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)


    def test_energies_singlepo_list(self):
        phase_shift = 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([1, 0, -1, 0, 1])

        WavePotential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude,
                                          y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential(wave_potentials=[WavePotential])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)


    def test_energies(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([2, 0,  -2, 0, 2])

        WavePotential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)


    def test_energies_phase_shifted(self):
        phase_shift1 = 0.0
        phase_shift2 = 180
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0, 90, 180, 270, 360]
        expected_result = np.array([0, 0, 0, 0, 0])

        WavePotential = pot.wavePotential(phase_shift=phase_shift1, multiplicity=multiplicity, amplitude=amplitude,
                                          y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential(phase_shift=phase_shift2, multiplicity=multiplicity, amplitude=amplitude,
                                           y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.ene(positions)

        self.assertEqual(type(expected_result), type(energies),
                         msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)



    def test_dHdpos(self):
        phase_shift= 0.0
        multiplicity = 1.0
        amplitude = 1.0
        y_offset = 0.0
        radians = False

        positions = [0,90, 180, 270, 360]
        expected_result = np.array([0, 2,  0, -2, 0])

        WavePotential = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        WavePotential2 = pot.wavePotential(phase_shift=phase_shift, multiplicity=multiplicity, amplitude=amplitude, y_offset=y_offset, radians=radians)
        potential = pot.torsionPotential(wave_potentials=[WavePotential, WavePotential2])
        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

class potentialCls_coulombPotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.coulombPotential()

    def test_energies(self):
        q1 = 1
        q2 = 1
        epsilon = 1

        positions = [0, 0.2, 0.5, 1, 2, 360]
        expected_result = np.array([np.inf, 0.397887358, 0.159154943,0.0795774715, 0.0397887358, 0.000221048532])

        potential = pot.coulombPotential(q1=q1, q2=q2, epsilon=epsilon)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)


    def test_dHdpos(self):
        q1 = 1
        q2 = 1
        epsilon = 1

        positions = [0, 0.2, 0.5, 1, 2, 360]
        expected_result = np.array([-np.inf, -1.98943679, -0.31830988, -0.0795774715, -0.0198943679, -0.000000614023700])

        potential = pot.coulombPotential(q1=q1, q2=q2, epsilon=epsilon)
        energies = potential.dhdpos(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=list(expected_result), actual=list(energies), err_msg="The results of "+potential.name+" are not correct!", decimal=8)

class potentialCls_lennardJonesPotential(unittest.TestCase):
    def test_constructor(self):
        potential = pot.lennardJonesPotential()

    def test_energies(self):
        c6: float = 1**(-1)
        c12: float = 1**(-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3 ,6 ]
        expected_result = np.array([np.nan,  9.99999000*10**11, 2.44125000*10**8, 4.03200000*10**3, 0, -1.53808594*10**-2, -1.36986044*10**-3, -2.14330111*10**-5])

        potential = pot.lennardJonesPotential(c6=c6, c12=c12, x_shift=x_shift, y_shift=y_shift)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=2)


    def test_dHdpos(self):
        c6: float = 1**(-1)
        c12: float = 1**(-1)
        x_shift: float = 0
        y_shift = 0

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3 ,6 ]
        expected_result = np.array([np.inf, 1.19999940*10**14, 1.46479687*10**10, 9.75360000*10**4, 6.00000000, -4.54101562*10**-2, -2.73595752*10**-3, -2.14325517*10**-5])

        potential = pot.lennardJonesPotential(c6=c6, c12=c12, x_shift=x_shift, y_shift=y_shift)
        energies = potential.dhdpos(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_allclose(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!")

class potentialCls_doubleWellPot1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.doubleWellPot()

    def test_energies(self):
        Vmax=100
        a=0
        b=8

        positions = np.linspace(-10, 10, num=5)
        expected_result = np.array([31.640625, 37.13378906, 100, 37.13378906, 31.640625])

        potential = pot.doubleWellPot(Vmax=Vmax, a=a, b=b)
        energies = potential.ene(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct!", decimal=2)

    def test_dHdpos(self):
        Vmax=100
        a=0
        b=8

        positions = [0, 0.1, 0.2, 0.5, 1, 2, 3, 6]
        expected_result = np.array([0, -0.62490234,  -1.24921875,  -3.11279297,  -6.15234375, -11.71875, -16.11328125, -16.40625])

        potential = pot.doubleWellPot(Vmax=Vmax, a=a, b=b)
        energies = potential.dhdpos(positions)

        print(energies)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of " + potential.name + " are not correct!",
                                       decimal=2)

class potentialCls_perturbedLinCoupledHosc(unittest.TestCase):
    def test_constructor(self):
        potential = pot.linCoupledHosc()

    def test_energies(self):
        ha = pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb = pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.linCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        expected_result = np.array([12.5, 0, 12.5, 50, 112.5])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies only for pot HB
        lam = 1
        expected_result = np.array([112.5, 50, 12.5, 0, 12.5])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([62.5, 25, 12.5, 25, 62.5])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)


    def test_dHdpos(self):
        ha=pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb=pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.linCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        #energies only for pot HA
        lam=0
        potential.set_lam(lam=lam)
        expected_result = np.array([-5, 0, 5, 10, 15])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies only for pot HB
        lam = 1
        expected_result = np.array([-15, -10, -5, 0, 5])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([-10, -5, 0, 5, 10])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)


    def test_dHdlam(self):
        ha = pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb = pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.linCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lam(lam=lam)
        expected_result = np.array([100, 50, 0, -50, -100])

        energies = potential.dhdlam(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

class potentialCls_perturbedExpCoupledHosc(unittest.TestCase):
    def test_constructor(self):
        potential = pot.expCoupledHosc()

    def test_energies(self):
        ha = pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb = pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.expCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        expected_result = np.array([12.5, 0, 12.5, 50, 112.5])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies only for pot HB
        lam = 1
        expected_result = np.array([112.5, 50, 12.5, 0, 12.5])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([12.78, 0.28, 12.5, 0.28,12.78])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)


    def test_dHdpos(self):
        ha=pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb=pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.expCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        #energies only for pot HA
        lam=0
        potential.set_lam(lam=lam)
        expected_result = np.array([3.60e-013, 0.00e+000, 3.60e-013, 3.43e-053, 1.52e-120])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies only for pot HB
        lam = 1
        expected_result = np.array([1.52218398e-120, 3.42854487e-053, 3.59703755e-013, 0.00, 3.59703755e-013])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([1.80e-13, 1.71e-53, 3.60e-13, 1.71e-53, 1.80e-13])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)


    def test_dHdlam(self):
        ha = pot.harmonicOsc(fc=1.0, x_shift=-5.0)
        hb = pot.harmonicOsc(fc=1.0, x_shift=5.0)
        potential = pot.expCoupledHosc(ha=ha, hb=hb, lam=0)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lam(lam=lam)
        expected_result = np.array([ 4.009080e-001,  4.009080e-001, 0.000, -5.846620e+053, -8.526387e+107])

        energies = potential.dhdlam(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_allclose(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies))

class potentialCls_perturbedHarmonicOsc1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.pertHarmonicOsc1D()

    def test_energies(self):
        fc = 1.0
        alpha = 1.0
        gamma = 0.0
        lam = 0
        potential = pot.pertHarmonicOsc1D(fc=fc, alpha=alpha, gamma=gamma, lam=lam)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lam(lam=lam)
        expected_result = np.array([50. , 12.5,  0. , 12.5, 50. ])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies only for pot HB
        lam = 1
        expected_result = np.array([100,  25, 0, 25, 100])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

        # energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([75, 18.75, 0, 18.75, 75.])

        potential.set_lam(lam=lam)
        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

    def test_dHdpos(self):
        fc = 1.0
        alpha = 1.0
        gamma = 0.0
        lam = 0
        potential = pot.pertHarmonicOsc1D(fc=fc, alpha=alpha, gamma=gamma, lam=lam)
        positions = np.linspace(-10, 10, num=5)

        #energies only for pot HA
        lam=0
        potential.set_lam(lam=lam)
        expected_result = np.array([-10, -5, 0, 5, 10])

        energies = potential.dhdpos(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies only for pot HB
        lam = 1
        expected_result = np.array([-20, -10, 0, 10, 20])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)

        #energies merged for pot HB and HA
        lam = 0.5
        expected_result = np.array([-15, -7.5, 0, 7.5, 15])

        potential.set_lam(lam=lam)
        energies = potential.dhdpos(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)


    def test_dHdlam(self):
        fc = 1.0
        alpha = 1.0
        gamma = 0.0
        lam = 0
        potential = pot.pertHarmonicOsc1D(fc=fc, alpha=alpha, gamma=gamma, lam=lam)

        positions = np.linspace(-10, 10, num=5)

        # energies only for pot HA
        lam = 0
        potential.set_lam(lam=lam)
        expected_result = np.array([50, 12.5, 0, 12.5, 50])

        energies = potential.dhdlam(positions)

        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

"""
class potentialCls_envelopedDoubleWellPotential1D(unittest.TestCase):
    def test_constructor(self):
        potential = pot.envelopedDoubleWellPotential1D()

    def test_energies(self):
        y_shifts= [0,0]
        x_shifts= [-5,5]
        smoothing= 1.0,
        fcs = [1,1]
        potential = pot.envelopedDoubleWellPotential1D(y_shifts=y_shifts, x_shifts=x_shifts,
                 smoothing=smoothing, fcs=fcs)

        positions = np.linspace(-10, 10, num=5)
        expected_result = np.array([-10, -5, 0, 5, 10])

        energies = potential.ene(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies,
                                       err_msg="The results of " + potential.name + " are not correct wit lambda " + str(
                                           lam) + "!\n\tPositions: " + str(positions) + "\n\tEnergies: " + str(energies), decimal=2)

    def test_dHdpos(self):
        y_shifts= [0,0]
        x_shifts= [-5,5]
        smoothing= 1.0,

        potential = pot.envelopedDoubleWellPotential1D()
        positions = np.linspace(-10, 10, num=5)

        expected_result = np.array([-10, -5, 0, 5, 10])

        energies = potential.dhdpos(positions)

        print(energies)
        print(positions)
        self.assertEqual(type(expected_result), type(energies), msg="returnType of potential was not correct! it should be an np.array")
        np.testing.assert_almost_equal(desired=expected_result, actual=energies, err_msg="The results of "+potential.name+" are not correct wit lambda "+str(lam)+"!\n\tPositions: "+str(positions)+"\n\tEnergies: "+str(energies), decimal=2)
"""

if __name__ == '__main__':
    unittest.main()
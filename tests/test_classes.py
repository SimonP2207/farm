import os
import unittest
from unittest.mock import patch
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt

from farm.sky_model.classes import SkyModel, SkyComponent
from farm.sky_model.tb_functions import gdsm2016_t_b
import farm.tests.fits_test_data as ftd

test_dcy = Path(os.path.dirname(__file__))
test_fits1_inu = test_dcy / 'test_fits1_inu.fits'
test_fits2_inu = test_dcy / 'test_fits2_inu.fits'
test_fits1_snu = test_dcy / 'test_fits1_snu.fits'
test_fits2_snu = test_dcy / 'test_fits2_snu.fits'
test_fits1_tb = test_dcy / 'test_fits1_tb.fits'
test_fits2_tb = test_dcy / 'test_fits2_tb.fits'


class Test_SkyComponent(unittest.TestCase):
    # This decorator allows the instantiation of an instance of an abstract
    # class (for testing purposes only!)
    @patch("farm.sky_model.classes._BaseSkyClass.__abstractmethods__", set())
    def setUp(self) -> None:
        self.sm1_inu = SkyComponent.load_from_fits(test_fits1_inu)
        self.sm2_inu = SkyComponent.load_from_fits(test_fits2_inu)
        self.sm1_snu = SkyComponent.load_from_fits(test_fits1_snu)
        self.sm2_snu = SkyComponent.load_from_fits(test_fits2_snu)
        self.sm1_tb = SkyComponent.load_from_fits(test_fits1_tb)
        self.sm2_tb = SkyComponent.load_from_fits(test_fits2_tb)
        self.gdsm = SkyComponent('GDSM 2016', (100, 100), 5. / 60,
                                 ftd.test_coord0, tb_func=gdsm2016_t_b)

    def test_load_from_fits(self):
        # 32-bit precision is to 7 decimal places, so set absolute tolerance to
        # 0.000001
        sms = (self.sm1_tb, self.sm1_snu, self.sm1_inu,
               self.sm2_tb, self.sm2_snu, self.sm2_inu)
        for sm in sms:
            if sm in (self.sm1_tb, self.sm1_inu, self.sm1_snu):
                npt.assert_allclose(sm.data('K'), ftd.test_tb1,
                                    rtol=1e-5)
                npt.assert_allclose(sm.data('JY/PIXEL'), ftd.test_snu1,
                                    rtol=1e-5)
                npt.assert_allclose(sm.data('JY/SR'), ftd.test_inu1,
                                    rtol=1e-5)
                self.assertEqual(sm.n_x, len(ftd.test_tb1[0][0]))
                self.assertEqual(sm.n_y, len(ftd.test_tb1[0]))
                self.assertEqual(len(sm.frequencies), len(ftd.test_tb1))

            elif sm in (self.sm2_tb, self.sm2_inu, self.sm2_snu):
                npt.assert_allclose(sm.data('K'), ftd.test_tb2,
                                    rtol=1e-5)
                npt.assert_allclose(sm.data('JY/PIXEL'), ftd.test_snu2,
                                    rtol=1e-5)
                npt.assert_allclose(sm.data('JY/SR'), ftd.test_inu2,
                                    rtol=1e-5)
                self.assertEqual(sm.n_x, len(ftd.test_tb2[0][0]))
                self.assertEqual(sm.n_y, len(ftd.test_tb2[0]))
                self.assertEqual(len(sm.frequencies), len(ftd.test_tb2))

            self.assertEqual(sm.header['NAXIS1'], 2)
            self.assertEqual(sm.header['NAXIS2'], 2)
            self.assertEqual(sm.header['NAXIS3'], 3)
            self.assertAlmostEqual(sm.header['CDELT1'], -ftd.test_cdelt)
            self.assertAlmostEqual(sm.header['CDELT2'], +ftd.test_cdelt)
            self.assertAlmostEqual(sm.header['CDELT3'],
                                   ftd.test_frequencies[1] -
                                   ftd.test_frequencies[0])
            self.assertAlmostEqual(sm.header['CRVAL1'], ftd.test_coord0.ra.deg)
            self.assertAlmostEqual(sm.header['CRVAL2'], ftd.test_coord0.dec.deg)
            self.assertAlmostEqual(sm.header['CRVAL3'], ftd.test_frequencies[0])

            self.assertAlmostEqual(sm.cdelt, ftd.test_cdelt)
            self.assertAlmostEqual(sm.coord0.ra.deg, ftd.test_coord0.ra.deg)
            self.assertAlmostEqual(sm.coord0.dec.deg, ftd.test_coord0.dec.deg)

    def test_add_frequency(self):
        self.gdsm.add_frequency(1.2e9)

        self.assertIsInstance(self.gdsm.frequencies, np.ndarray)
        self.assertEqual(self.gdsm.frequencies, [1.2e9])

        self.gdsm.add_frequency(2.2e9)
        np.testing.assert_array_equal(self.gdsm.frequencies, np.array([1.2e9, 2.2e9]))

        with self.assertWarns(UserWarning):
            self.gdsm.add_frequency([4.2e9, 3.2e9, [3.2e9], 2.2e9, [1.2e9]])

        np.testing.assert_array_equal(self.gdsm.frequencies, [1.2e9, 2.2e9, 3.2e9, 4.2e9])

    def test_frequency_present(self):
        self.gdsm.add_frequency(1.2e9)
        self.assertTrue(self.gdsm.frequency_present(1.2e9))

        self.gdsm.add_frequency([0.2e9])

        self.assertEqual(self.gdsm.frequency_present(0.2e9), 0.2e9)
        self.assertEqual(self.gdsm.frequency_present(1.2e9), 1.2e9)

        with self.assertRaises(TypeError):
            self.gdsm.frequency_present([1e9, 2e9])

    def test_header(self):
        self.gdsm.add_frequency(1.2e9)
        self.assertEqual(self.gdsm.header['CDELT3'], 1.)
        self.assertEqual(self.gdsm.header["CRVAL3"], 1.2e9)

        self.gdsm.add_frequency([4.2e9, 3.2e9, 2.2e9, 0.2e9])
        self.assertEqual(self.gdsm.header["NAXIS3"], 5)
        self.assertEqual(self.gdsm.header["CRVAL3"], 0.2e9)
        self.assertEqual(self.gdsm.header["CDELT3"], 1.e9)

    def test_frequencies(self):
        np.testing.assert_array_equal(self.gdsm.frequencies,
                                      [])
        self.gdsm.add_frequency([1.3e9, 4.3e9])
        np.testing.assert_array_equal(self.gdsm.frequencies,
                                      [1.3e9, 4.3e9])

        self.gdsm.add_frequency([3.3e9, 2.3e9])
        np.testing.assert_array_equal(self.gdsm.frequencies,
                                      [1.3e9, 2.3e9, 3.3e9, 4.3e9])


class Test_SkyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(Test_SkyModel, cls).setUpClass()
        cls.sm1_inu = SkyComponent.load_from_fits(test_fits1_inu)
        cls.sm2_inu = SkyComponent.load_from_fits(test_fits2_inu)
        cls.sm1_snu = SkyComponent.load_from_fits(test_fits1_snu)
        cls.sm2_snu = SkyComponent.load_from_fits(test_fits2_snu)
        cls.sm1_tb = SkyComponent.load_from_fits(test_fits1_tb)
        cls.sm2_tb = SkyComponent.load_from_fits(test_fits2_tb)

    def setUp(self) -> None:
        self.sky_model_inu = SkyModel((len(ftd.test_tb2[0][0]),
                                       len(ftd.test_tb2[0])), ftd.test_cdelt,
                                      ftd.test_coord0, ftd.test_frequencies)
        self.sky_model_snu = SkyModel((len(ftd.test_tb2[0][0]),
                                       len(ftd.test_tb2[0])), ftd.test_cdelt,
                                      ftd.test_coord0, ftd.test_frequencies)
        self.sky_model_tb = SkyModel((len(ftd.test_tb2[0][0]),
                                      len(ftd.test_tb2[0])), ftd.test_cdelt,
                                     ftd.test_coord0, ftd.test_frequencies)

    def test_add_component(self):
        self.sky_model_inu.add_component(self.sm1_inu)
        self.sky_model_inu.add_component(self.sm2_inu)
        self.sky_model_snu.add_component(self.sm1_snu)
        self.sky_model_snu.add_component(self.sm2_snu)
        self.sky_model_tb.add_component(self.sm1_tb)
        self.sky_model_tb.add_component(self.sm2_tb)
        npt.assert_allclose(self.sky_model_inu.data('JY/SR'),
                            ftd.test_inu1 + ftd.test_inu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_snu.data('JY/PIXEL'),
                            ftd.test_snu1 + ftd.test_snu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_tb.data('K'),
                            ftd.test_tb1 + ftd.test_tb2, rtol=1e-5)

    def test___add__(self):
        self.sky_model_snu = self.sky_model_snu + self.sm1_snu
        self.sky_model_snu += self.sm2_snu
        self.sky_model_inu = self.sky_model_inu + self.sm1_inu
        self.sky_model_inu += self.sm2_inu
        self.sky_model_tb = self.sky_model_tb + self.sm1_tb
        self.sky_model_tb += self.sm2_tb

        npt.assert_allclose(self.sky_model_inu.data('JY/SR'),
                            ftd.test_inu1 + ftd.test_inu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_snu.data('JY/PIXEL'),
                            ftd.test_snu1 + ftd.test_snu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_tb.data('K'),
                            ftd.test_tb1 + ftd.test_tb2, rtol=1e-5)

    def test_data(self):
        self.sky_model_snu = self.sky_model_snu + self.sm1_snu
        self.sky_model_snu += self.sm2_snu
        self.sky_model_inu = self.sky_model_inu + self.sm1_inu
        self.sky_model_inu += self.sm2_inu
        self.sky_model_tb = self.sky_model_tb + self.sm1_tb
        self.sky_model_tb += self.sm2_tb

        npt.assert_allclose(self.sky_model_inu.data('JY/SR'),
                            ftd.test_inu1 + ftd.test_inu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_snu.data('JY/PIXEL'),
                            ftd.test_snu1 + ftd.test_snu2, rtol=1e-5)
        npt.assert_allclose(self.sky_model_tb.data('K'),
                            ftd.test_tb1 + ftd.test_tb2, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()

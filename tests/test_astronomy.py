import unittest

import numpy as np

import farm.astronomy as ast


REL_TOL = 1e-5


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tb = 2.12  # K
        self.freq = 2.4e8  # Hz
        self.int = 3751.7182582459272  # Jy/sr
        self.cdelt = 1.5 / 60. ** 2.  # deg
        self.omega = np.radians(self.cdelt) ** 2.  # sr
        self.flux = 1.984095027072151e-07  # Jy

    def test_gaussian_beam_area(self):
        bmaj = 1.0 / 3600.  # 1.0 arcsec
        bmin = 0.5 / 3600.  # 0.5 arcsec
        self.assertAlmostEqual(ast.gaussian_beam_area(bmaj, bmin),
                               1.331631801646914e-11,
                               delta=1.331631801646914e-11 * REL_TOL)

        with self.assertRaises(TypeError) as _:
            ast.gaussian_beam_area([bmaj], bmin)
        with self.assertRaises(TypeError) as _:
            ast.gaussian_beam_area(bmaj, [bmin])
        with self.assertRaises(TypeError) as _:
            ast.gaussian_beam_area([bmaj], [bmin])

    def test_tb_to_intensity(self):
        self.assertAlmostEqual(ast.tb_to_intensity(self.tb, self.freq),
                               self.int, delta=self.int * REL_TOL)
        self.assertAlmostEqual(ast.tb_to_intensity([self.tb], [self.freq]),
                               self.int, delta=self.int * REL_TOL)
        with self.assertRaises(ValueError) as _:
            ast.tb_to_intensity(self.tb, [self.freq])

    def test_intensity_to_tb(self):
        self.assertAlmostEqual(ast.intensity_to_tb(self.int, self.freq),
                               self.tb, delta=self.tb * REL_TOL)
        self.assertAlmostEqual(ast.intensity_to_tb([self.int], [self.freq]),
                               self.tb, delta=self.tb * REL_TOL)
        with self.assertRaises(ValueError) as _:
            ast.tb_to_intensity(self.int, [self.freq])

    def test_tb_to_flux(self):
        self.assertAlmostEqual(ast.tb_to_flux(self.tb, self.freq, self.omega),
                               self.flux, delta=self.flux * REL_TOL)
        self.assertAlmostEqual(ast.tb_to_flux([self.tb], [self.freq], self.omega),
                               self.flux, delta=self.flux * REL_TOL)
        with self.assertRaises(ValueError) as _:
            ast.tb_to_flux(self.tb, [self.freq], self.omega)

    def test_flux_to_tb(self):
        self.assertAlmostEqual(ast.flux_to_tb(self.flux, self.freq, self.omega),
                               self.tb, delta=self.tb * REL_TOL)
        self.assertAlmostEqual(ast.flux_to_tb([self.flux], [self.freq], self.omega),
                               self.tb, delta=self.tb * REL_TOL)
        with self.assertRaises(ValueError) as _:
            ast.flux_to_tb(self.flux, [self.freq], self.omega)


if __name__ == '__main__':
    unittest.main()

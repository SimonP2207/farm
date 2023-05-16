import unittest
import warnings

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import farm.physics.astronomy as ast
from farm.miscellaneous import decorators

REL_TOL = 1e-5


# def filter_erfa_warnings(func):
#     """Decorator to filter out cryptic ErfaWarnings thrown by erfa library"""
#     def decorator(*args, **kwargs):
#         """decorator function"""
#         from erfa import ErfaWarning
#
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")#, ErfaWarning)
#             return func(*args, **kwargs)
#     return func


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

    @decorators.suppress_warnings("astropy", "erfa", "ephem")
    def test_lst_to_utc(self):
        date = Time("2027-01-01 16:18:39.165")
        lon = 123.45
        lst = 4.4

        midnight = Time(date.strftime("%Y-%m-%d 00:00:00"))
        lst_midnight = midnight.sidereal_time('apparent',
                                              longitude=lon * u.deg).hourangle

        correct_answer = midnight + (lst + (24. - lst_midnight)) * u.hour
        answer = ast.lst_to_utc(lst, date, lon)
        self.assertEqual(correct_answer.strftime("%Y-%m-%d %H:%M:%S"),
                         answer.strftime("%Y-%m-%d %H:%M:%S"))

        lst = 17.4
        correct_answer = midnight + (lst - lst_midnight) * u.hour
        answer = ast.lst_to_utc(lst, date, lon)
        self.assertEqual(correct_answer.strftime("%Y-%m-%d %H:%M:%S"),
                         answer.strftime("%Y-%m-%d %H:%M:%S"))

    @decorators.suppress_warnings("astropy", "erfa", "ephem")
    def test_scan_times(self):
        loc = EarthLocation.from_geodetic(lon=116.6311 * u.deg,  # ASKAP
                                          lat=-26.697 * u.deg)
        t0 = Time("2027-05-22 15:32:00", location=loc)
        coord = SkyCoord("23:04:12.47", "-45:43:12.2",
                         unit=(u.hourangle, u.deg), frame='fk5')
        min_elevations = np.linspace(10., 90., 9)
        min_gap_scan = 800
        t_totals = np.arange(3600 * 2, 3600 * 12 + 1, 7200)
        n_scans = np.arange(5, 15, 1)

        for min_el in min_elevations:
            t_up = 2. * ast.ha_at_elevation(coord, loc.lat.deg, min_el)
            t_up *= ast.sidereal_ratio * u.hour

            # TODO: 'partial_scans_allowed = True' not tested
            for t_total in t_totals:
                for n_scan in n_scans:
                    t_scan = t_total / n_scan * u.s
                    msg = ("testing farm.physics.astronomy.scan_times with "
                           "following values:\n"
                           f"\tt0:             {t0.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"\tcoord:          {coord.to_string('hmsdms')}\n"
                           f"\tloc:            Lon {loc.lon.deg:.3f}deg, Lat {loc.lat.deg:.3f}deg\n"
                           f"\tn_scan:         {n_scan}\n"
                           f"\tt_total:        {t_total}s\n"
                           f"\tmin_el:         {min_el:.1f}deg\n"
                           f"\tmin_gap_scan:   {min_gap_scan}s\n"
                           f"\tpartial scans?: {True}")
                    if t_up < t_scan:
                        with self.assertRaises(ValueError):
                            ast.scan_times(t0, coord, loc, n_scan, t_total,
                                           min_el, min_gap_scan)
                    else:
                        scans = ast.scan_times(t0, coord, loc, n_scan, t_total,
                                               min_el, min_gap_scan)
                        t_scans = sum([e - s for s, e in scans]).to_value('s')
                        self.assertAlmostEqual(
                            t_scans, t_total, delta=1,
                            msg=f"len(t_scans) ({t_scans:0f})s != t_total "
                                f"whilst {msg}"
                        )
                        self.assertEqual(
                            len(scans), n_scan,
                            msg=f"len(n_scans) ({len(n_scans)}) != n_scan "
                                f"whilst {msg}"
                        )




if __name__ == '__main__':
    unittest.main()

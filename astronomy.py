"""
Astronomy-related methods and classes
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta


def get_start_time(ra0_deg, length_sec):
    """Returns optimal start time for field RA and observation length."""
    t = Time('2021-09-21 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = 24.0 - t.sidereal_time('apparent').hour + (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start


def get_dt_days(ra0_deg):
    """Returns optimal centre time for field RA."""
    t = Time('2021-09-21 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_days = (24.0 - t.sidereal_time('apparent').hour + (ra0_deg / 15.0)) / 24.
    return dt_days


def bright_sources():
    """Returns a list of bright A-team sources."""
    # Sgr A: removed since will use Haslam instead
    # For A: data from the Molonglo Southern 4 Jy sample (VizieR).
    # Others from GLEAM reference paper, Hurley-Walker et al. (2017), Table 2.
    return np.array((
        [50.67375, -37.20833, 528, 0, 0, 0, 178e6, -0.51, 0, 0, 0, 0],  # For
        [201.36667, -43.01917, 1370, 0, 0, 0, 200e6, -0.50, 0, 0, 0, 0],  # Cen
        [139.52500, -12.09556, 280, 0, 0, 0, 200e6, -0.96, 0, 0, 0, 0],  # Hyd
        [79.95833, -45.77889, 390, 0, 0, 0, 200e6, -0.99, 0, 0, 0, 0],  # Pic
        [252.78333, 4.99250, 377, 0, 0, 0, 200e6, -1.07, 0, 0, 0, 0],  # Her
        [187.70417, 12.39111, 861, 0, 0, 0, 200e6, -0.86, 0, 0, 0, 0],  # Vir
        [83.63333, 22.01444, 1340, 0, 0, 0, 200e6, -0.22, 0, 0, 0, 0],  # Tau
        [299.86667, 40.73389, 7920, 0, 0, 0, 200e6, -0.78, 0, 0, 0, 0],  # Cyg
        [350.86667, 58.81167, 11900, 0, 0, 0, 200e6, -0.41, 0, 0, 0, 0]  # Cas
    ))


def angle_to_galactic_plane(coord: SkyCoord) -> float:
    skypos1 = SkyCoord(ra=coord.ra, dec=(coord.dec.deg + 0.5) * u.degree)

    l0, b0 = coord.galactic.l.degree, coord.galactic.b.degree
    l1, b1 = skypos1.galactic.l.degree, skypos1.galactic.b.degree

    return np.arctan2((l1 - l0), (b1 - b0))

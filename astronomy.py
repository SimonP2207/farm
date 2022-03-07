"""
Astronomy-related methods and classes
"""
from typing import Union, Tuple
import numpy.typing as npt
import numpy as np
import scipy.constants as con
from astropy.io.fits import Header
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from powerbox.tools import get_power


def power_spectrum(header: Header, data: np.ndarray,
                   sum_freq: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dims_deg = (header["NAXIS1"] * np.abs(header["CDELT1"]),
                header["NAXIS2"] * header["CDELT2"])

    if sum_freq:
        ps_data = get_power(np.nanmean(data, axis=0), boxlength=dims_deg,
                            ignore_zero_mode=True,
                            vol_normalised_power=True)[::-1]
    else:
        assert(data.ndim == 3, "Something has gone wrong with the datas' shape")
        scales, power = [], []
        for idx in range(data):
            scales, power_nu = get_power(data[idx], boxlength=dims_deg,
                                         ignore_zero_mode=True,
                                         vol_normalised_power=True)[::-1]
            power = np.append(power, power_nu)
        ps_data = scales, power

    return ps_data


def gaussian_beam_area(bmaj: float, bmin: float) -> float:
    """
    Area of a Gaussian beam [sr]

    Parameters
    ----------
    bmaj
        Beam major axis FWHM [deg]
    bmin
        Beam minor axis FWHM [deg]

    Returns
    -------
    Total area under 2D Gaussian
    """
    if not isinstance(bmaj, float):
        raise TypeError(f"bmaj must be a float, not {type(bmaj)}")
    if not isinstance(bmin, float):
        raise TypeError(f"bmin must be a float, not {type(bmin)}")

    bmin_rad, bmaj_rad = np.radians(bmin), np.radians(bmaj)

    return np.pi * bmaj_rad * bmin_rad / (4. * np.log(2.))


def intensity_to_tb(intensity: Union[float, npt.ArrayLike],
                    freq: Union[float, npt.ArrayLike]
                    ) -> Union[float, npt.ArrayLike]:
    """
    Convert intensity (Jy/sr) to brightness temperature (K)

    Parameters
    ----------
    intensity
        Intensity [Jy/Sr]
    freq
        Frequency at which to compute intensity [Hz]

    Returns
    -------
    Brightness temperature [K]
    """
    ndims = np.ndim(intensity)
    if np.shape(freq) == np.shape(intensity):
        pass
    elif not np.isscalar(freq):
        if ndims == 0 or not 2 <= ndims <= 3:
            raise ValueError("Must be either 2 or 3 dimensions")

    wls = con.speed_of_light / (freq if np.isscalar(freq) else np.asarray(freq))

    if ndims == 3 and not np.isscalar(wls):
        wls = wls[:, np.newaxis, np.newaxis]

    return ((wls ** 2. / (2. * con.Boltzmann)) *
            ((intensity if np.isscalar(intensity) else np.asarray(intensity)) *
             1e-26))


def intensity_to_flux(intensity: Union[float, npt.ArrayLike],
                      solid_angle: float) -> Union[float, npt.ArrayLike]:
    """
    Convert intensity (Jy/sr) to flux (Jy/pixel or Jy/beam)

    Parameters
    ----------
    intensity
        Intensity [Jy/sr]

    solid_angle
        Solid angle of flux region e.g. pixel size, beam [sr]

    Returns
    -------
    Flux [Jy]
    """
    return intensity * solid_angle


def flux_to_tb(flux: Union[float, npt.ArrayLike],
               freq: Union[float, npt.ArrayLike],
               solid_angle: float) -> Union[float, npt.ArrayLike]:
    """
    Convert intensity (Jy/sr) to brightness temperature (K)

    Parameters
    ----------
    flux
        Flux [Jy]
    freq
        Frequency at which to compute intensity [Hz]
    solid_angle
        Solid angle of flux region e.g. pixel size, beam [sr]

    Returns
    -------
    Brightness temperature [K]
    """
    ndims = np.ndim(flux)
    if np.shape(freq) == np.shape(flux):
        pass
    elif not np.isscalar(freq):
        if ndims == 0 or not 2 <= ndims <= 3:
            raise ValueError("Must be either 2 or 3 dimensions")

    wls = con.speed_of_light / (freq if np.isscalar(freq) else np.asarray(freq))

    if ndims == 3 and not np.isscalar(wls):
        wls = wls[:, np.newaxis, np.newaxis]

    return ((wls ** 2. / (2. * con.Boltzmann * solid_angle)) *
            ((flux if np.isscalar(flux) else np.asarray(flux)) *
             1e-26))


def flux_to_intensity(flux: Union[float, npt.ArrayLike],
                      solid_angle: float) -> Union[float, npt.ArrayLike]:
    """
    Convert flux (Jy/pixel or Jy/beam) to intensity (Jy/sr)

    Parameters
    ----------
    flux
        Flux [Jy]

    solid_angle
        Solid angle of flux region e.g. pixel size, beam [sr]

    Returns
    -------
    Intensity [Jy/sr]
    """
    return flux / solid_angle


def tb_to_intensity(t_b: Union[float, npt.ArrayLike],
                    freq: Union[float, npt.ArrayLike]
                    ) -> Union[float, npt.ArrayLike]:
    """
    Convert brightness temperature (K) to intensity (Jy/sr)

    Parameters
    ----------
    t_b
        Brightness temperature [K].
    freq
        Frequency at which to compute intensity [Hz]

    Returns
    -------
    Intensity [Jy/Sr]
    """
    ndims = np.ndim(t_b)
    if np.shape(freq) == np.shape(t_b):
        pass
    elif not np.isscalar(freq):
        if ndims == 0 or not 2 <= ndims <= 3:
            raise ValueError("t_b must be either 2 or 3 dimensions")

    wls = con.speed_of_light / (freq if np.isscalar(freq) else np.asarray(freq))

    if ndims == 3 and not np.isscalar(wls):
        wls = wls[:, np.newaxis, np.newaxis]

    return (((2. * con.Boltzmann) / wls ** 2.) *
            (t_b if np.isscalar(t_b) else np.asarray(t_b))) / 1e-26


def tb_to_flux(t_b: Union[float, npt.ArrayLike],
               freq: Union[float, npt.ArrayLike],
               solid_angle: float) -> Union[float, npt.ArrayLike]:
    """
    Convert brightness temperature (K) to intensity (Jy/sr)

    Parameters
    ----------
    t_b
        Brightness temperature [K].
    freq
        Frequency at which to compute intensity [Hz]
    solid_angle
        Solid angle of flux region e.g. pixel size, beam [sr]

    Returns
    -------
    Intensity [Jy/Sr]
    """
    ndims = np.ndim(t_b)
    if np.shape(freq) == np.shape(t_b):
        pass
    elif not np.isscalar(freq):
        if ndims == 0 or not 2 <= ndims <= 3:
            raise ValueError("Must be either 2 or 3 dimensions")

    wls = con.speed_of_light / (freq if np.isscalar(freq) else np.asarray(freq))

    if ndims == 3 and not np.isscalar(wls):
        wls = wls[:, np.newaxis, np.newaxis]

    return (((2. * con.Boltzmann * solid_angle) / wls ** 2.) *
            (t_b if np.isscalar(t_b) else np.asarray(t_b))) / 1e-26


def get_start_time(ra0_deg: float, length_sec: float) -> Time:
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

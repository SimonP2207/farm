"""
Astronomy-related methods and classes
"""
from typing import Union, Tuple, List
import numpy.typing as npt
import numpy as np
import scipy.constants as con
from astropy.io.fits import Header
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from astropy.coordinates.angles import Longitude, Latitude, Angle

from ..miscellaneous import decorators

sidereal_day = 23. * u.hour + 56. * u.minute + 4.1 * u.s
sidereal_ratio = (sidereal_day / (24. * u.hour)).value


def power_spectrum(header: Header, data: np.ndarray,
                   sum_freq: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    from powerbox.tools import get_power

    dims_deg = (header["NAXIS1"] * np.abs(header["CDELT1"]),
                header["NAXIS2"] * header["CDELT2"])

    if sum_freq:
        ps_data = get_power(np.nanmean(data, axis=0), boxlength=dims_deg,
                            ignore_zero_mode=True,
                            vol_normalised_power=True)[::-1]
    else:
        assert data.ndim == 3, "Something has gone wrong with the datas' shape"
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
    Total area under 2D Gaussian [sr]
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


def angle_to_galactic_plane(coord: SkyCoord) -> float:
    skypos1 = SkyCoord(ra=coord.ra, dec=(coord.dec.deg + 0.5) * u.degree)

    l0, b0 = coord.galactic.l.degree, coord.galactic.b.degree
    l1, b1 = skypos1.galactic.l.degree, skypos1.galactic.b.degree

    return np.arctan2((l1 - l0), (b1 - b0))


def angular_separation(
        ra0: float, dec0: float, ra1: Union[float, npt.ArrayLike],
        dec1: Union[float, npt.ArrayLike]
) -> Union[float, npt.ArrayLike]:
    """
    Calculates angular separation between two coordinates on the sky

    Parameters
    ----------
    ra0
        Right ascension of coordinate to measure from [deg]
    dec0
        Declination of coordinate to measure from [deg]
    ra1
        Right ascension(s) of coordinate to measure to [deg]
    dec1
        Declination(s) of coordinate to measure to [deg]

    Returns
    -------
    Separation(s) from (ra0, dec0) to (ra1, dec1) [deg]
    """
    ra0_rad, dec0_rad = np.radians(ra0), np.radians(dec0)
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)

    angle = np.arccos(np.sin(dec0_rad) * np.sin(dec1_rad) +
                      np.cos(dec0_rad) * np.cos(dec1_rad) *
                      np.cos(ra0_rad - ra1_rad))

    return np.degrees(angle)


def position_angle(
        ra0: float, dec0: float, ra1: Union[float, npt.ArrayLike],
        dec1: Union[float, npt.ArrayLike], positive: bool = True
) -> Union[float, npt.ArrayLike]:
    """
    Calculates position angle (east from north) between two coordinates on the
    sky

    Parameters
    ----------
    ra0
        Right ascension of coordinate to measure from [deg]
    dec0
        Declination of coordinate to measure from [deg]
    ra1
        Right ascension(s) of coordinate to measure to [deg]
    dec1
        Declination(s) of coordinate to measure to [deg]
    positive
        Whether to return position angle in the range 0 < angle < 360 (True,
        default), or -180 < angle < 180 (False)

    Returns
    -------
    Position angle(s) from (ra0, dec0) to (ra1, dec1) [deg]
    """
    ra0_rad, dec0_rad = np.radians(ra0), np.radians(dec0)
    ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
    dra_rad = ra1_rad - ra0_rad

    angle = np.arctan2(np.sin(dra_rad),
                       np.cos(dec0_rad) * np.tan(dec1_rad) -
                       np.sin(dec0_rad) * np.cos(dra_rad))

    if positive:
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    return np.degrees(angle)


def within_square_fov(
        fov: tuple[float, float], ra0: float, dec0: float,
        ra1: Union[float, npt.ArrayLike],
        dec1: Union[float, npt.ArrayLike]
) -> Union[float, npt.ArrayLike]:
    """
    Calculates position angle (east from north) between two coordinates on the
    sky

    Parameters
    ----------
    fov
        Field of view as a 2-tuple (fov_x, fov_y) [deg]
    ra0
        Right ascension of field centre [deg]
    dec0
        Declination of field centre [deg]
    ra1
        Right ascension(s) of coordinate in question [deg]
    dec1
        Declination(s) of coordinate in question [deg]

    Returns
    -------
    Position angle(s) from (ra0, dec0) to (ra1, dec1) [deg]
    """

    sep = angular_separation(ra0, dec0, ra1, dec1)
    angle = position_angle(ra0, dec0, ra1, dec1, positive=True)

    angle = np.where((90 < angle) & (angle < 180), 180 - angle, angle)
    angle = np.where((180 < angle) & (angle < 270), angle - 180, angle)
    angle = np.where((270 < angle) & (angle < 360), 360 - angle, angle)

    ddec = np.abs(dec1 - dec0)
    dra = np.abs(sep * np.sin(np.radians(angle)))

    return (dra < fov[0] / 2) & (ddec < fov[1] / 2)


@decorators.suppress_warnings("astropy", "erfa", "ephem")
def elevation_at_utc(coord: SkyCoord, utc: Time,
                     location: EarthLocation) -> Time:
    """
    Calculate the elevation of a celestial coordinate at a given UTC, from
    a specific location on the Earth

    Parameters
    ----------
    coord
        Celestial coordinate
    utc
        UTC time atn which to compute the elevation
    location
        Earth location at which to compute elevation
    Returns
    -------
    Elevation at UTC
    """
    frame = AltAz(obstime=utc, location=location)
    alts_azs = coord.transform_to(frame)

    return alts_azs.alt


def utc_zenith(coord: SkyCoord, date: Time):
    """
    UTC time at which a celestial coordinate as at its zenith

    Parameters
    ----------
    coord
        Celestial coordinate
    date
        Date on which to compute the UTC

    Returns
    -------
    Time instance for the UTC
    """
    return date - date.sidereal_time('apparent').value * u.hour


def utc_range_above_elevation(
        minimum_el: float,
        coord: SkyCoord,
        date: Time,
        location: EarthLocation
) -> Union[None, Tuple[Time, Time]]:
    """
    Calculate the time range during which a celestial coordinate is located
    above a minimum elevation on the night sky, from a specific earth
    location

    Parameters
    ----------
    minimum_el
        Minimum elevation of the coordinate [deg
    coord
        Coordinate of the target
    date
        Date on which to calculate the UTC range
    location
        Earth location

    Returns
    -------
    2-tuple of  (rise time, set time) or None if never above elevation
    """
    utc_max_el = utc_zenith(coord, date)
    dhs = np.linspace(0, 12, 13) * u.hour
    dms = np.linspace(0, 60, 61) * u.minute
    dss = np.linspace(0, 60, 61) * u.s
    # TODO: What if it rises/falls within an hour?

    for i_h, dh in enumerate(dhs):
        el = elevation_at_utc(coord,
                              utc_max_el + dh,
                              location)
        if el < minimum_el * u.deg:
            h = dhs[i_h - 1]
            for i_m, dm in enumerate(dms):
                el = elevation_at_utc(coord,
                                      utc_max_el + h + dm,
                                      location)
                if el < minimum_el * u.deg:
                    m = dms[i_m - 1]
                    for i_s, ds in enumerate(dss):
                        el = elevation_at_utc(coord,
                                              utc_max_el + h + m + ds,
                                              location)
                        if el < minimum_el * u.deg:
                            s = dss[i_s - 1]
                            fall_time = utc_max_el + h + m + s
                            rise_time = utc_max_el - (fall_time - utc_max_el)
                            return rise_time, fall_time
    return None


def in_lst_range(lst, lst_range):
    if lst < 0. * u.hourangle or lst >= 24. * u.hourangle:
        raise ValueError('lst must be in range 0h - 24h')
    if lst_range[0] < lst_range[1]:
        if lst_range[0] < lst < lst_range[1]:
            return True
    else:
        if 24 * u.hourangle > lst > lst_range[0]:
            return True
        elif 0. * u.hourangle < lst < lst_range[1]:
            return True
    return False


def ha_at_utc(utc, coord, loc: Union[None, EarthLocation]):
    """

    Parameters
    ----------
    utc
    coord
    lat

    Returns
    -------

    """
    if utc.location is not None:
        lst = utc.sidereal_time('apparent')
    else:
        if not isinstance(loc, EarthLocation):
            raise TypeError(f"Provided utc does not have attached location,"
                            f"therefore must define loc as EarthLocation "
                            f"instance, not {loc}")
        lst = utc.sidereal_time('apparent', longitude=loc)

    lst_zenith = coord.ra.to('hourangle')

    return lst - lst_zenith


def ha_at_elevation(coord, lat, elev):
    """
    Given a celestial coordinate and observatory latitude, calculate the
    celestial coordinate's hour angle at a given elevation.
    Parameters
    ----------
    coord: astropy.coordinates.SkyCoord
        Object's RA/DEC
    lat: float
        Observatory latitude in degrees
    elev: float
        Elevation to compute hour angle at in degrees
    Returns
    -------
    Hour angle as an astropy.coordinates.angles.Longitude instance in units
    of hours
    """
    assert isinstance(coord, SkyCoord)
    assert isinstance(lat, float)
    assert isinstance(elev, float)

    if elev < -90. or elev > 90.:
        raise ValueError("elev must be in range -90 <= el <= +90, "
                         "not {}".format(elev))

    if lat < -90. or lat > 90.:
        raise ValueError("lat must be in range -90 <= lat <= +90,"
                         "not {}".format(lat))

    lat_angle = Latitude(lat, unit=u.deg)
    el_angle = Angle(elev * u.deg)
    ra, dec = coord.ra, coord.dec

    p1 = np.sin(el_angle.rad) - np.sin(lat_angle.rad) * np.sin(dec.rad)
    p2 = np.cos(lat_angle.rad) * np.cos(dec.rad)

    ha = Longitude(np.arccos(p1 / p2) * u.rad).hourangle

    if np.isnan(ha):
        return 0.

    return ha


@decorators.suppress_warnings("astropy", "erfa", "ephem")
def scan_times(t0: Time, coord_target: SkyCoord, telescope_loc: EarthLocation,
               n_scan: int, t_total: int, min_el: float = 20.0,
               min_gap_scan: int = 0,
               partial_scans_allowed: bool = False) -> List[Tuple[Time, Time]]:
    """
    Get scan times ranges for an observation

    Parameters
    ----------
    t0
        Start time of first scan
    coord_target
        Celestial coordinate of target
    telescope_loc
        Location of telescope
    n_scan
        Number of scans
    t_total
        Total time on source [s]
    min_el
        Minimum elevation [deg]. Default is 20.0
    min_gap_scan
        Minimum time-gap between consecutive scans [s]. Default is 0
    partial_scans_allowed
        Whether to break a scan at an elevation boundary or not. If True, add
        the remaining time of the partial scan to the next scan. If False,
        maintain number of scans and equal scan times. Default is False

    Returns
    -------
    List of n_scan * 2-tuples containing (scan_start_time, scan_end_time)

    Raises
    ------
    ValueError
        If the source does not have enough time above the minimum elevation,
        such that requested total on-source time and number of scans are
        incompatible
    """
    # Hour-angle of source at time of setting below minimum elevation
    ha_set = ha_at_elevation(coord_target, telescope_loc.lat.deg, min_el)
    ha_set *= u.hourangle
    scan_length = (t_total / n_scan) * u.s  # Desired scan length
    t_up = 2. * ha_set.value * u.hour * sidereal_ratio  # Total time above minimum elevation per day
    t_down = sidereal_day - t_up  # Total time below minimum elevation per day

    if t_up < scan_length:
        raise ValueError(f"Source only above {min_el:.1f}deg for "
                         f"{t_up:.2f}hr, but scan length required for "
                         f"{n_scan:.0f} scans is "
                         f"{scan_length.to_value('h'):.2f} hr. "
                         f"Reduce scan length by increasing number of scans")

    # LST times for coordinate's rise, zenith and set
    lst_zenith = coord_target.ra.to('hourangle')
    lst_set = lst_zenith + ha_set  # LST source-set time
    lst_set = lst_set % (24. * u.hourangle)
    lst_rise = lst_zenith - ha_set
    lst_rise = lst_rise % (24. * u.hourangle)

    # Initialise loop variables
    t_start = t0 + 0 * u.s  # Copy of desired start time
    t_over = 0 * u.s  # Time under minimum elevation during previous scan
    n_partial_scans = 0  # Number of partial scans conducted
    scans = []  # Hold scan times

    # Run loop
    while len(scans) - n_partial_scans < n_scan:
        el_t_start = elevation_at_utc(coord_target, t_start, telescope_loc)
        lst_start = t_start.sidereal_time('apparent')

        if el_t_start < min_el * u.deg:
            if lst_start > lst_rise:
                t_start += t_down - ((lst_start - lst_set).value *
                                     sidereal_ratio * u.hour)
            else:
                t_start += ((lst_rise - lst_start).value *
                            sidereal_ratio * u.hour)
            lst_start = t_start.sidereal_time('apparent')

        t_end = t_start + scan_length + t_over
        el_t_end = elevation_at_utc(coord_target, t_end, telescope_loc)
        t_over = 0. * u.s
        if el_t_end < min_el * u.deg:
            if not partial_scans_allowed:
                t_start += t_down - ((lst_start - lst_set).value *
                                     sidereal_ratio * u.hour)
                continue

            lst_end = t_end.sidereal_time('apparent')
            t_over = (lst_end - lst_set).value * u.hour
            t_end -= t_over
            n_partial_scans += 1

        if len(scans) != 0:
            t_on_source = sum([e - s for s, e in scans]).to_value('s')
        else:
            t_on_source = 0

        if t_on_source >= t_total:
            break
        elif t_on_source + (t_end - t_start).to_value('s') > t_total:
            new_t_total = t_on_source + (t_end - t_start).to_value('s')
            t_end -= (new_t_total - t_total) * u.s

        scans.append((t_start, t_end))
        t_start = t_end + min_gap_scan * u.s

    return scans


def lst_to_utc(lst: float, date: Time, lon: float) -> Time:
    """
    Calculate the UTC for a given LST and date

    Parameters
    ----------
    lst
        Local Sidereal Time required [hour-angle]
    date
        Date on which to convert from LST to UTC
    lon
        Longitude of telescope [deg]

    Returns
    -------

    """
    midnight = Time(date.strftime("%Y-%m-%d 00:00:00"))
    lst_midnight = midnight.sidereal_time('apparent',
                                          longitude=lon * u.deg).hourangle

    if lst_midnight > lst:
        dh = 24. - lst_midnight + lst
    else:
        dh = lst - lst_midnight

    return midnight + dh * u.hour

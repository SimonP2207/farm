"""
Module containing all methods which return the sky brightness temperature
distribution for a variety of different foreground components and science
case-related components.

Any function-additions to this module must adhere to the following conventions:
- Function naming should be '*whatever_name_you_choose*_t_b'
- Function argument #1 must be a farm.classes.SkyComponent instance
- Function argument #2 must be a frequency or array of frequencies
- Returned value must be a numpy array of dtype=np.float32
"""
from typing import Union, Protocol, TypeVar, Callable
import warnings
import pathlib
import numpy as np
import numpy.typing as npt
from reproject import reproject_from_healpix
from astropy.coordinates import SkyCoord

if __name__ == '__main__':
    from farm.physics import astronomy as ast
else:
    from ..physics import astronomy as ast


# Define expected types for typing
SkyComponent = TypeVar('SkyComponent')
FreqType = Union[float, npt.NDArray[np.float32]]
ReturnType = npt.NDArray[np.float32]


class TbFunction(Protocol):
    """
    Typing class to give correct arg and return types for brightness temperature
    functions
    """
    def __call__(self, sky_component: SkyComponent,
                 freq: FreqType) -> ReturnType: ...


def gdsm2016_t_b(sky_component: SkyComponent, freq: FreqType) -> ReturnType:
    """
    Brightness temperature function for global diffuse sky model

    Parameters
    ----------
    sky_component
        SkyComponent instance
    freq
        Frequency(s) at which to calculate the sky temperature brightness
        distribution [Hz]

    Returns
    -------
    2D brightness temperature distribution [K]
    """
    import logging
    from pygdsm import GlobalSkyModel2016
    from astropy.io import fits
    from farm.miscellaneous import generate_random_chars

    # NOTE: Tested data_unit='TRJ' (T_b) and data_unit='MJsysr' (I_nu) and both
    # give same results with apropriate farm-code below
    gdsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='MJysr',
                              resolution='hi')
    # gdsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='TRJ',
    #                           resolution='hi')
    temp_fitsfile = pathlib.Path(f'temp{generate_random_chars(10)}.fits')

    gdsm.generate(freq)

    # Catches warnings thrown by healpy.fitsfunc.write_map which is used by the
    # BaseSkyModel of pygdsm.base_skymodel. These warnings are "setting the
    # output map dtype to [dtype('float32')]" and thrown because BaseSkyModel
    # does not provide healpy's write_map function a dtype.
    # warnings.simplefilter doesn't work to suppress the warnings as the
    # warnings are raised by healpy's Logger instance
    logging.getLogger("healpy").setLevel(logging.CRITICAL)  # raise Logger level
    gdsm.write_fits(str(temp_fitsfile))  # expects str type
    logging.getLogger("healpy").setLevel(logging.WARNING)  # reset Logger level

    # User context manager here to automatically open/close file
    with fits.open(temp_fitsfile) as hdugsm:
        hdugsm[1].header.set('COORDSYS', 'G')
        i_nu = np.single(reproject_from_healpix(hdugsm[1],
                                                sky_component.header2d))[0]
        i_nu *= 1e6  # MJy/sr -> Jy/sr
    temp_fitsfile.unlink()  # Remove temporary .fits file

    return ast.intensity_to_tb(i_nu, freq)


def fits_t_b(sky_component: SkyComponent, freq: FreqType) -> ReturnType:
    """
    Brightness temperature function for a .fits file

    Parameters
    ----------
    sky_component
        SkyComponent instance
    freq
        Frequency(s) at which to calculate the sky temperature brightness
        distribution [Hz]

    Returns
    -------
    2D brightness temperature distribution [K]
    """
    matching_freq_idx = np.where(sky_component.frequencies == freq)[0][0]

    return sky_component.data('K')[matching_freq_idx]


def padovani_core_t_b(core_type: str, b_field: int, dist_pc: float,
                      core_coord: SkyCoord) -> Callable:
    """
    Wrapper to produce brightness temperature function for a Padovani molecular
    core which produces non-thermal, synchrotron emission. See Padovani & Galli
    (2018) for details

    Parameters
    ----------
    core_type
        Type of core density profile, one of 'A', 'B', or 'C' for peaked,
        intermediate and flat cases, respectively
    b_field
        Magnetic field in uG. One of 50, 100, or 200
    dist_pc
        Distance to core [pc]
    core_coord
        Central coordinate of core as an astropy.coordinates.SkyCoord instance

    Returns
    -------
    Brightness temperature function for case specified by core_type and b_field
    """
    from farm.data import DATA_FILE_DCY

    core_type = core_type.upper()
    if isinstance(b_field, float):
        b_field = round(b_field)

    assert b_field in (50, 100, 200)
    assert core_type in ('A', 'B', 'C')

    padovani_data_file = DATA_FILE_DCY / 'PadovaniCore.npz'

    def func(sky_component: SkyComponent, freq: FreqType) -> ReturnType:
        f"""
        Brightness temperature function for a Padovani molecular core case 
        '{core_type}' which produces non-thermal, synchrotron emission with a
        magnetic field of {b_field}uG. See Padovani & Galli (2018) for details

        Parameters
        ----------
        sky_component
            SkyComponent instance
        freq
            Frequency(s) at which to calculate the sky temperature brightness
            distribution [Hz]

        Returns
        -------
        2D brightness temperature distribution [K]
        """

        from astropy.wcs import WCS
        import scipy.constants as con

        with np.load(padovani_data_file, 'rt') as f:
            padovani_nus = f.get('nu')
            padovani_rs = f.get('R') / 100.
            padovani_i_rs = f.get(f"I_{core_type}{b_field}") * 1e-3 / 1e-26

        def neighbour_idxs(arr, value):
            """
            Modified binary search tree returning indices of neighbouring values
            if value inserted into the correct location within a sorted arr
            """
            # Lower and upper bounds
            start, end = 0, len(arr) - 1

            # Traverse the search space
            while start <= end:
                mid = (start + end) // 2
                if arr[mid] == value:
                    return mid, mid + 1
                elif arr[mid] < value:
                    start = mid + 1
                else:
                    end = mid - 1

            # Return the insert position
            return end, end + 1

        def get_i_r_at_freq(nu: float):
            """Interpolate I(r) at a given frequency"""
            freq_in_nus = np.isclose(nu, padovani_nus, atol=1)

            # Return i_r if frequency exists
            if any(freq_in_nus):
                nu_idx = np.squeeze(np.where(freq_in_nus))
                return padovani_i_rs[nu_idx]

            idxs = list(neighbour_idxs(padovani_nus, freq))

            # Interpolate if frequency lies in range given by Padovani data cube
            # and extrapolate if frequency lies outside range given in Padovani
            # data cube
            if idxs[0] < 0:
                idxs = [_ + 1 for _ in idxs]
            elif idxs[1] >= len(padovani_nus):
                idxs = [_ - 1 for _ in idxs]
            elif idxs[0] >= 0 and idxs[1] < len(padovani_nus):
                pass
            else:
                raise ValueError(f"Something has gone wrong: idxs = {idxs}")

            # Catch divide by 0 warnings etc.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                m = ((np.log10(padovani_i_rs[idxs[1]]) -
                      np.log10(padovani_i_rs[idxs[0]])) /
                     (np.log10(padovani_nus[idxs[1]]) -
                      np.log10(padovani_nus[idxs[0]])))

                c = np.log10(padovani_i_rs[idxs[1]]) - \
                    m * np.log10(padovani_nus[idxs[1]])
                extrap_i_r = 10 ** (m * np.log10(nu) + c)

            return np.nan_to_num(extrap_i_r, nan=0., posinf=0., neginf=0.)

        wcs = WCS(sky_component.header2d)
        yy, xx = np.meshgrid(np.arange(1, sky_component.header['NAXIS2'] + 1),
                             np.arange(1, sky_component.header['NAXIS1'] + 1),
                             indexing='ij')

        core_coord_pix = wcs.wcs_world2pix(core_coord.ra.deg,
                                           core_coord.dec.deg, 1)

        # Calculate dr (distance from centre of core) coordinate system
        xx_dx, yy_dy = xx - core_coord_pix[0], yy - core_coord_pix[1]
        cdelt = sky_component.header['CDELT2']  # deg
        m_per_pix = dist_pc * con.parsec * np.tan(np.radians(cdelt))
        xx_dr, yy_dr = xx_dx * m_per_pix, yy_dy * m_per_pix
        dr = np.sqrt(xx_dr ** 2. + yy_dr ** 2.)

        def intensity(i):
            """Intensity as a function of r"""
            def func_(dr_):
                """Returns intensity at dr_"""
                zz = np.zeros_like(dr_)
                # Extract 2d-slice of dr within the bounds of padovani_rs
                min_x_idx, min_y_idx = np.min(
                    np.where(dr_ < np.max(padovani_rs)), axis=1
                )
                max_x_idx, max_y_idx = np.max(
                    np.where(dr_ < np.max(padovani_rs)), axis=1
                )

                min_x_idx -= 1 if min_x_idx != 0 else 0
                min_y_idx -= 1 if min_y_idx != 0 else 0
                max_x_idx += 1 if max_x_idx != len(dr_) else 0
                max_y_idx += 1 if max_y_idx != len(dr_) else 0

                dr__ = dr_[min_x_idx:max_x_idx, min_y_idx:max_y_idx]

                # Cases where dr__ == 0. are dealt with below
                idxs = np.searchsorted(padovani_rs, dr__).flatten()

                # Get intensity values at insertion index, zz1, and also before
                # it, zz2
                zz1 = np.take_along_axis(np.append(i, 0), idxs, axis=0)
                zz2 = np.take_along_axis(np.append(i, 0), idxs - 1, axis=0)

                drr1 = np.take_along_axis(np.append(padovani_rs, np.nan),
                                          idxs, axis=0)
                drr2 = np.take_along_axis(np.append(padovani_rs, np.nan),
                                          idxs - 1, axis=0)

                # m = (zz2 - zz1) / (idxs - (idxs - 1))
                # c = zz2 - m * idxs
                m = (zz2 - zz1) / (drr2 - drr1)
                c = zz2 - m * drr2

                zzs = m * dr__.flatten() + c

                ints = np.where(idxs < len(padovani_rs), zzs, 0.)
                ints = np.reshape(ints, np.shape(dr__))

                # In case dr__ is ever 0.
                ints = np.where(dr__ != 0., ints, i[0])
                zz[min_x_idx:max_x_idx, min_y_idx:max_y_idx] = ints

                return np.nan_to_num(zz, nan=0., posinf=0., neginf=0.)

            return func_

        i_rs = get_i_r_at_freq(freq)
        if isinstance(freq, float):
            intensities = intensity(i_rs)(dr)
        else:
            intensities = [intensity(i_r)(dr) for i_r in i_rs]

        return ast.intensity_to_tb(intensities, freq)

    return func

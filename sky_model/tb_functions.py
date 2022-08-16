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
from typing import Union, Protocol, TypeVar
import pathlib
import numpy as np
import numpy.typing as npt
from reproject import reproject_from_healpix
from farm.physics import astronomy as ast

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
    Brightness temperature function for a .fits image cube

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

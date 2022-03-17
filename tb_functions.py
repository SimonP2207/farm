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
from . import astronomy as ast

# Define expected types for typing
SkyCompType = TypeVar('SkyComponent')
FreqType = Union[float, npt.NDArray[np.float32]]
ReturnType = npt.NDArray[np.float32]


class TbFunction(Protocol):
    def __call__(self,sky_component: SkyCompType,
                 freq: FreqType) -> ReturnType: ...


def gdsm2016_t_b(sky_component: SkyCompType, freq: FreqType) -> ReturnType:
    from pygdsm import GlobalSkyModel2016
    from astropy.io import fits
    from .miscellaneous import generate_random_chars

    gdsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='MJysr',
                              resolution='hi')

    temp_fitsfile = pathlib.Path(f'temp{generate_random_chars(10)}.fits')

    gdsm.generate(freq)
    gdsm.write_fits(str(temp_fitsfile))  # expects str type

    # User context manager here to automatically open/close file
    with fits.open(temp_fitsfile) as hdugsm:
        hdugsm[1].header.set('COORDSYS', 'G')
        i_nu = np.single(reproject_from_healpix(hdugsm[1],
                                                sky_component.header2d))[0]
        i_nu *= 1e6  # MJy/sr -> Jy/sr

    temp_fitsfile.unlink()  # Remove temporary .fits file

    return ast.intensity_to_tb(i_nu, freq)


def fits_t_b(sky_component: SkyCompType, freq: FreqType) -> ReturnType:
    matching_freq_idx = np.where(sky_component.frequencies == freq)[0][0]

    return sky_component.data('K')[matching_freq_idx]

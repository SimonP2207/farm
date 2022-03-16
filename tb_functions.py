from typing import Union
import pathlib
import numpy as np
import numpy.typing as npt
from reproject import reproject_from_healpix
from . import astronomy as ast


def gdsm2016_t_b(
        sky_component: 'SkyComponent',
        freq: Union[float, npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    from pygdsm import GlobalSkyModel2016
    from astropy.io import fits
    from .classes import generate_random_chars

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


def fits_t_b(sky_component: 'SkyComponent',
             freq: Union[float, npt.ArrayLike]) -> npt.NDArray[np.float32]:
    return sky_component.data('K')[np.where(sky_component.frequencies == freq)[0][0]]

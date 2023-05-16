"""
Contains any file-handling methods related to .fits files
"""
from pathlib import Path
from typing import Tuple

import astropy.coordinates
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.io.fits import Header

from . import error_handling as errh
from .decorators import ensure_is_fits


def fits_bunit(fitsfile: Path) -> str:
    """
    Get brightness unit from .fits header. If 'BUNIT' not present in .fits
    header, best guess the data unit
    """
    header, image_data = fits_hdr_and_data(fitsfile)

    if 'BUNIT' in header:
        return header["BUNIT"].strip().upper()
    else:
        return None


def fits_equinox(fitsfile: Path, raise_warnings: bool = False) -> float:
    """Get equinox from .fits header. Assume J2000 if absent"""

    header, _ = fits_hdr_and_data(fitsfile)

    try:
        return header["EQUINOX"]
    except KeyError:
        # Assume J2000 if information not present in header
        if raise_warnings:
            errh.issue_warning(
                UserWarning, "Equinox information not present. Assuming J2000"
            )
        return 2000.


def fits_hdr_and_data(fitsfile: Path) -> Tuple[Header, npt.NDArray]:
    """Return header and data from a .fits image/cube"""
    with fits.open(fitsfile) as hdulist:
        return hdulist[0].header, hdulist[0].data


def fits_frequencies(fitsfile: Path) -> npt.NDArray:
    """Get list of frequencies of a .fits cube, as np.ndarray"""
    header, _ = fits_hdr_and_data(fitsfile)
    return fits_hdr_frequencies(header)


def fits_hdr_frequencies(header: Header) -> npt.NDArray:
    """Get list of frequencies from a .fits header, as np.ndarray"""
    freq_min = (header["CRVAL3"] - (header["CRPIX3"] - 1) *
                header["CDELT3"])
    freq_max = (header["CRVAL3"] + (header["NAXIS3"] - header["CRPIX3"]) *
                header["CDELT3"])
    return np.linspace(freq_min, freq_max, header["NAXIS3"])


def hdr2d(n_x: int, n_y: int, coord0: astropy.coordinates.SkyCoord,
          cdelt: float) -> Header:
    """
    Create a 2D (RA and Dec) header

    Parameters
    ----------
    n_x
        Number of pixels in x/R.A.
    n_y
        Number of pixels in y/declination
    coord0
        Central pixel coordinate
    cdelt
        Cell size [deg]

    Returns
    -------
    Created header as an astropy.io.fits.Header instance
    """
    hdr = Header({'Simple': True})
    hdr.set('BITPIX', -32)
    hdr.set('NAXIS', 2)
    hdr.set('NAXIS1', n_x)
    hdr.set('NAXIS2', n_y)
    hdr.set('CTYPE1', 'RA---SIN')
    hdr.set('CTYPE2', 'DEC--SIN')
    hdr.set('CRVAL1', coord0.ra.deg)
    hdr.set('CRVAL2', coord0.dec.deg)
    hdr.set('CRPIX1', n_x / 2)
    hdr.set('CRPIX2', n_y / 2)
    hdr.set('CDELT1', -cdelt)
    hdr.set('CDELT2', cdelt)
    hdr.set('CUNIT1', 'deg     ')
    hdr.set('CUNIT2', 'deg     ')
    hdr.set('EQUINOX', {'fk4': 1950., 'fk5': 2000.}[coord0.frame.name])

    return hdr


def hdr3d(n_x: int, n_y: int, coord0: astropy.coordinates.SkyCoord,
          cdelt: float, frequencies: npt.ArrayLike) -> Header:
    """
    Create a 3D (RA, declination and frequency) header

    Parameters
    ----------
    n_x
        Number of pixels in x/R.A.
    n_y
        Number of pixels in y/declination
    coord0
        Central pixel coordinate
    cdelt
        Cell size [deg]
    frequencies
        Full list/numpy array of cube channel frequencies

    Returns
    -------
    Created header as an astropy.io.fits.Header instance
    """
    hdr = hdr2d(n_x, n_y, coord0, cdelt)
    hdr.insert('NAXIS2', ('NAXIS3', len(frequencies)), after=True)
    hdr.insert('CTYPE2', ('CTYPE3', 'FREQ    '), after=True)
    hdr.insert('CRVAL2', ('CRVAL3', min(frequencies)), after=True)
    hdr.insert('CRPIX2', ('CRPIX3', 1), after=True)
    hdr.insert('CDELT2', ('CDELT3', 1.), after=True)
    hdr.insert('CUNIT2', ('CUNIT3', 'Hz      '), after=True)

    if len(frequencies) > 1:
        hdr.set('CDELT3', frequencies[1] - frequencies[0])

    return hdr


def is_fits(file: Path) -> bool:
    """Determine if a file is a fits file"""
    try:
        with fits.open(file) as _:
            return True
    except OSError:
        return False


@ensure_is_fits('fits_file')
def is_fits_table(fits_file: Path) -> bool:
    """Determine if a fits file is a table"""
    if not is_fits(fits_file):
        return False

    with fits.open(fits_file) as hdul:
        return any([hasattr(_, 'columns') for _ in hdul])


@ensure_is_fits('fits_file')
def is_fits_image(fits_file: Path) -> bool:
    """Determine if a fits file is an image"""
    if not is_fits(fits_file):
        return False

    return not is_fits_table(fits_file)


@ensure_is_fits('fits_table')
def fits_table_to_dataframe(fits_table: Path) -> pd.DataFrame:
    """Convert fits table to pandas dataframe"""
    with fits.open(fits_table) as hdulist:
        return hdulist_to_dataframe(hdulist)


def hdulist_to_dataframe(hdulist):
    """Convert astropy.fits.HDUList to pandas.DataFrame"""
    if sum([hasattr(_, 'columns') for _ in hdulist]) > 1:
        errh.issue_warning("WARNING",
                           "More than one table present in .fits. Will "
                           "return the first table in HDUList")
    for hdu in hdulist:
        if hasattr(hdu, 'columns'):
            return pd.DataFrame.from_records(hdu.data)

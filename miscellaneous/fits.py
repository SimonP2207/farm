from pathlib import Path
from astropy.io import fits
import pandas as pd
from . import error_handling as errh
from .decorators import ensure_is_fits


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

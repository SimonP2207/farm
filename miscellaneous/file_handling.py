"""
Contains any file-handling methods not related to .fits files
"""
import pathlib
import pandas as pd
import numpy as np


def is_osm_table(osmfile: pathlib.Path) -> bool:
    """
    Determine if a text file is a legitimate oskar sky model file i.e. it has 12
    columns of data

    Parameters
    ----------
    osmfile
        Full path to oskar file

    Returns
    -------
    True if a legitimate oskar sky model file with 12 data columns, False
    otherwise
    """
    data = np.loadtxt(osmfile)

    if np.shape(data)[1] == 12:
        return True

    return False


def osm_to_dataframe(osmfile: pathlib.Path) -> pd.DataFrame:
    """
    Create a pandas DataFrame instance from an oskar sky model file

    Parameters
    ----------
    osmfile
        Full path to oskar sky model file

    Returns
    -------
    pandas.DataFrame instance
    """
    from . import error_handling as errh

    if not is_osm_table(osmfile):
        errh.raise_error(TypeError, f"{osmfile} is not an oskar sky model")

    loaded_data = np.loadtxt(osmfile)
    columns = ['ra', 'dec', 'fluxI', 'fluxQ', 'fluxU', 'fluxV',
               'freq0', 'spix', 'rm', 'maj', 'min', 'pa']

    return pd.DataFrame(data=loaded_data, columns=columns)

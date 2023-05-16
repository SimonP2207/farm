"""
Contains any file-handling methods not related to .fits files
"""
import pathlib

import pandas
import pandas as pd
import numpy as np

_OSM_COLUMNS = ['ra', 'dec', 'fluxI', 'fluxQ', 'fluxU', 'fluxV',
               'freq0', 'spix', 'rm', 'maj', 'min', 'pa']


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

    with open(osmfile, 'rt') as f:
        loaded_data = np.loadtxt(f)

    return pd.DataFrame(data=loaded_data, columns=_OSM_COLUMNS)


def dataframe_to_osm(df: pandas.DataFrame, filename: pathlib.Path):
    """Write a dataframe to file in the oskar sky model format"""
    mandatory_cols = ('ra', 'dec', 'fluxI', 'freq0')
    non_mandatory_cols = ('spix', 'maj', 'min', 'pa', 'rm',
                          'fluxQ', 'fluxU', 'fluxV')

    mandatory_cols_present = [col in df.columns for col in mandatory_cols]
    if not all(mandatory_cols_present):
        missing_col = mandatory_cols[mandatory_cols_present.index(False)]
        raise ValueError(f"'{missing_col}' column not present in DataFrame, "
                         f"which contains the columns, {df.columns}")

    df_copy = df.copy(deep=True)
    for col in non_mandatory_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0.

    df_copy.drop(
        columns=[col for col in df_copy.columns if col not in _OSM_COLUMNS]
    )

    with open(filename, 'wt') as f:
        np.savetxt(f, df_copy[_OSM_COLUMNS].values,
                   fmt=('%.10f', '%.10f', '%.10f', '%.10f', '%.10f', '%.10f',
                        '%.3f', '%.3f', '%.1f', '%.1f', '%.1f', '%.1f'))

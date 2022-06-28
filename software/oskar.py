"""
Methods/classes related to the command-line interface with OSKAR
"""
import io
import subprocess
import pathlib
from typing import Union, TextIO, Optional
from functools import partial

from astropy.time import Time
import pandas as pd
from oskar import *

from . import common as sfuncs
from ..miscellaneous import error_handling as errh

# Default column names for OSKAR sky model files
oskar_sky_model_cols = {'ra': 'ra_deg', 'dec': 'dec_deg',
                        'fluxI': 'I', 'fluxQ': 'Q', 'fluxU': 'U',
                        'fluxV': 'V', 'freq0': 'ref_freq_hz',
                        'spix': 'spectral_index', 'rm': 'rotation_measure',
                        'maj': 'major_axis_arcsec',
                        'min': 'minor_axis_arcsec',
                        'pa': 'position_angle_deg'}

if sfuncs.which('oskar') is None:
    errh.raise_error(ImportError, "oskar is not in your PATH")

oskar_path = pathlib.Path(sfuncs.which('oskar'))


def add_dataframe_to_sky(df: pd.DataFrame, sky: Sky,
                         col_dict: Optional[dict] = None):
    """
    Add all entrys in a pandas.DataFrame to an oskar.Sky instance

    Parameters
    ----------
    df
        pandas.DataFrame
    sky
        oskar.Sky to append DataFrame sources to
    col_dict
        Dictionary containing df column names as the values, with the
        following keys/format:

            {
                'ra': 'right ascension column name',  # str only
                'dec': 'right ascension column name',  # str only
                'fluxI': 'Stokes-I flux column name',  # str only
                'fluxQ': 'Stokes-Q flux column name',  # str or None
                'fluxU': 'Stokes-U flux column name',  # str or None
                'fluxV': 'Stokes-V flux column name',  # str or None
                'freq0': 'reference frequency column name',  # str or None
                'spix': 'spectral index column name',  # str or None
                'rm': 'rotation measure column name',  # str or None
                'maj': 'FWHM major axis column name',  # str or None
                'min': 'FWHM minor axis column name',  # str or None
                'pa': 'position angle column name'  # str or None
            }

        In the case whereby col_dict is None (default), it is assumed that the
        DataFrame has all 12 required oskar sky model columns, appropriately
        named

    Returns
    -------
    None
    """
    if not col_dict:
        col_dict = {k: k for k in oskar_sky_model_cols.keys()}

    df_to_oskar_dict = {}
    for k, v in col_dict.items():
        if v is not None:
            df_to_oskar_dict[oskar_sky_model_cols[k]] = df[v]

    sky.append_sources(**df_to_oskar_dict)


def set_oskar_task_ini(
        oskar_task: str,
        ini_file_obj: TextIO, setting: str,
        value: Union[str, bool, float, int, Time, pathlib.Path]
) -> subprocess.CompletedProcess:
    if not isinstance(ini_file_obj, io.IOBase):
        errh.raise_error(TypeError,
                         "Please ensure 'ini_file' is a file object, "
                         f"not {type(ini_file_obj)}")

    value_str = ""
    if isinstance(value, str):
        value_str = value
    elif isinstance(value, bool):
        value_str = str(value).lower()
    elif isinstance(value, float):
        value_str = format(value, '.5f')
    elif isinstance(value, int):
        value_str = str(value)
    elif isinstance(value, Time):
        value_str = f'"{value.value}"'
    elif isinstance(value, pathlib.Path):
        value_str = f'"{str(pathlib.Path(value).resolve())}"'
    else:
        errh.raise_error(TypeError,
                         "Please ensure value is a valid type, not "
                         f"{type(value)}")

    file_name = f'"{str(pathlib.Path(ini_file_obj.name).resolve())}"'

    cmd = f"{sfuncs.which(oskar_task)} --set {{}} {{}} {{}}"

    return subprocess.run(cmd.format(file_name, setting, value_str),
                          shell='True')


def run_oskar_task(oskar_task: str, ini_file: Union[pathlib.Path, str]):
    if not isinstance(ini_file, pathlib.Path):
        if isinstance(ini_file, str):
            ini_file = pathlib.Path(ini_file)
        else:
            errh.raise_error(TypeError,
                             "Please ensure ini_file is either a Path instance "
                             f"or str, not {type(ini_file)}")

    if not ini_file.exists():
        errh.raise_error(FileNotFoundError, f"{ini_file.name} does not exist")

    cmd = f"{sfuncs.which(oskar_task)} {{}}"
    subprocess.run(cmd.format(str(ini_file.resolve())), shell='True')


# Parse list of available oskar tasks and set up relevant module variables to
# handle setting/running those oskar tasks
# This works, but not advistable as import inspections raise errors when any
# functions created below are imported
# for f in oskar_path.parent.glob('./*'):
#     if f.name.startswith('oskar_') and not f.is_dir():
#         for i, purpose in enumerate(('set', 'run')):
#             operation = (set_oskar_task_ini, run_oskar_task)[i]
#             locals()[f"{purpose}_{f.name}"] = partial(operation, f.name)

set_oskar_sim_beam_pattern = partial(set_oskar_task_ini,
                                     "oskar_sim_beam_pattern")
set_oskar_binary_file_query = partial(set_oskar_task_ini,
                                      "oskar_binary_file_query")
set_oskar_convert_ecef_to_enu = partial(set_oskar_task_ini,
                                        "oskar_convert_ecef_to_enu")
set_oskar_filter_sky_model_clusters = partial(set_oskar_task_ini,
                                              "oskar_filter_sky_model_clusters")
set_oskar_fits_image_to_sky_model = partial(set_oskar_task_ini,
                                            "oskar_fits_image_to_sky_model")
set_oskar_system_info = partial(set_oskar_task_ini,
                                "oskar_system_info")
set_oskar_vis_add_noise = partial(set_oskar_task_ini,
                                  "oskar_vis_add_noise")
set_oskar_vis_to_ms = partial(set_oskar_task_ini,
                              "oskar_vis_to_ms")
set_oskar_convert_geodetic_to_ecef = partial(set_oskar_task_ini,
                                             "oskar_convert_geodetic_to_ecef")
set_oskar_fit_element_data = partial(set_oskar_task_ini,
                                     "oskar_fit_element_data")
set_oskar_imager = partial(set_oskar_task_ini,
                           "oskar_imager")
set_oskar_sim_interferometer = partial(set_oskar_task_ini,
                                       "oskar_sim_interferometer")
set_oskar_vis_add = partial(set_oskar_task_ini,
                            "oskar_vis_add")
set_oskar_vis_summary = partial(set_oskar_task_ini,
                                "oskar_vis_summary")

run_oskar_sim_beam_pattern = partial(run_oskar_task,
                                     "oskar_sim_beam_pattern")
run_oskar_binary_file_query = partial(run_oskar_task,
                                      "oskar_binary_file_query")
run_oskar_convert_ecef_to_enu = partial(run_oskar_task,
                                        "oskar_convert_ecef_to_enu")
run_oskar_filter_sky_model_clusters = partial(run_oskar_task,
                                              "oskar_filter_sky_model_clusters")
run_oskar_fits_image_to_sky_model = partial(run_oskar_task,
                                            "oskar_fits_image_to_sky_model")
run_oskar_system_info = partial(run_oskar_task,
                                "oskar_system_info")
run_oskar_vis_add_noise = partial(run_oskar_task,
                                  "oskar_vis_add_noise")
run_oskar_vis_to_ms = partial(run_oskar_task,
                              "oskar_vis_to_ms")
run_oskar_convert_geodetic_to_ecef = partial(run_oskar_task,
                                             "oskar_convert_geodetic_to_ecef")
run_oskar_fit_element_data = partial(run_oskar_task,
                                     "oskar_fit_element_data")
run_oskar_imager = partial(run_oskar_task,
                           "oskar_imager")
run_oskar_sim_interferometer = partial(run_oskar_task,
                                       "oskar_sim_interferometer")
run_oskar_vis_add = partial(run_oskar_task,
                            "oskar_vis_add")
run_oskar_vis_summary = partial(run_oskar_task,
                                "oskar_vis_summary")

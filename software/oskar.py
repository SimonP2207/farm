"""
Methods/classes related to the command-line interface with OSKAR
"""
import io
import subprocess
import pathlib
from typing import Union, TextIO

from astropy.time import Time
from oskar import *

from . import common as sfuncs
from ..miscellaneous import error_handling as errh


if sfuncs.which('oskar') is None:
    errh.raise_error(ImportError, "oskar is not in your PATH")


def set_oskar_sim_beam_pattern(
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

    cmd = f"{sfuncs.which('oskar_sim_beam_pattern')} --set {{}} {{}} {{}}"

    return subprocess.run(cmd.format(file_name, setting, value_str),
                          shell='True')


def run_oskar_sim_beam_pattern(ini_file: Union[pathlib.Path, str]):

    if not isinstance(ini_file, pathlib.Path):
        if isinstance(ini_file, str):
            ini_file = pathlib.Path(ini_file)
        else:
            errh.raise_error(TypeError,
                             "Please ensure ini_file is either a Path instance "
                             f"or str, not {type(ini_file)}")

    if not ini_file.exists():
        errh.raise_error(FileNotFoundError, f"{ini_file.name} does not exist")

    cmd = f"{sfuncs.which('oskar_sim_beam_pattern')} {{}}"
    subprocess.run(cmd.format(str(ini_file.resolve())), shell='True')

"""
Methods/classes related to the command-line interface with OSKAR
"""

import pathlib, io, subprocess
from typing import Union

from astropy.time import Time

from errorhandling import raise_error


def set_oskar_sim_beam_pattern(ini_file_obj: Union[io.IOBase], setting: str,
                               value: Union[str, bool, float, int, Time, pathlib.Path]) -> subprocess.CompletedProcess:
    if not isinstance(ini_file_obj, io.IOBase):
        raise_error(TypeError,
                    "Please ensure 'ini_file' is a file object, "
                    f"not {type(ini_file_obj)}")

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
        value_str = f'"{pathlib.Path(value).resolve().__str__()}"'
    else:
        raise_error(TypeError,
                    f"Please ensure value is a vaild type, not {type(value)}")

    file_name = f'"{pathlib.Path(ini_file_obj.name).resolve().__str__()}"'

    cmd = '/usr/local/bin/oskar_sim_beam_pattern --set {} {} {}'
    return subprocess.run(cmd.format(file_name, setting, value_str),
                          shell='True')


def run_oskar_sim_beam_pattern(ini_file: Union[pathlib.Path, str]):
    print(isinstance(ini_file, pathlib.Path))
    if not isinstance(ini_file, pathlib.Path):
        if isinstance(ini_file, str):
            ini_file = pathlib.Path(ini_file)
        else:
            raise_error(TypeError,
                        "Please ensure ini_file is either a Path instance or "
                        f"str, not {type(ini_file)}")

    if not ini_file.exists():
        raise_error(FileNotFoundError, f"{ini_file.name} does not exist")

    cmd = '/usr/local/bin/oskar_sim_beam_pattern {}'
    subprocess.run(cmd.format(ini_file.__str__()), shell='True')

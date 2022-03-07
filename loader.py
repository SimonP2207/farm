import typing
import pathlib
import datetime
import toml

from farm import LOGGER
import errorhandling as errh


def check_file_exists(filename: pathlib.Path):
    if not filename.exists():
        errh.raise_error(FileNotFoundError,
                         f"{filename.resolve().__str__()} doesn't exist")


def check_config_validity(config_dict: dict):
    """Check .toml configuration file has all required parameters"""
    structure = (('directories', 'base', str),
                 ('directories', 'sky_models', str),
                 ('directories', 'telescope_models', str),
                 ('observation', 'time', datetime.datetime),
                 ('observation', 'field', 'ra0', str),
                 ('observation', 'field', 'dec0', str),
                 ('observation', 'field', 'frame', str),
                 ('observation', 'field', 'nxpix', int),
                 ('observation', 'field', 'nypix', int),
                 ('observation', 'field', 'cdelt', float),
                 ('observation', 'correlator', 'freq_min', float),
                 ('observation', 'correlator', 'freq_max', float),
                 ('observation', 'correlator', 'nchan', int),
                 ('observation', 'correlator', 'chanwidth', float),
                 ('sky_models', '21cm', 'include', bool),
                 ('sky_models', '21cm', 'create', bool),
                 ('sky_models', '21cm', 'image', str),
                 ('sky_models', 'A-Team', 'include', bool),
                 ('sky_models', 'GDSM', 'include', bool),
                 ('sky_models', 'GDSM', 'create', bool),
                 ('sky_models', 'GDSM', 'image', str),
                 ('sky_models', 'GSSM', 'include', bool),
                 ('sky_models', 'GSSM', 'create', bool),
                 ('sky_models', 'GSSM', 'image', str),
                 ('sky_models', 'PS', 'include', bool),
                 ('sky_models', 'PS', 'create', bool),
                 ('sky_models', 'PS', 'image', str),)

    for param in structure:
        entry = config_dict.copy()
        for i, level in enumerate(param[:-1]):
            try:
                entry = entry[level]
            except KeyError:
                errh.raise_error(KeyError,
                                 f"{'.'.join(param[:i + 1])} "
                                 "not present in config")

        if not isinstance(entry, param[-1]):
            errh.raise_error(TypeError,
                             f"{'.'.join(param[:-1])} incorrect type")

    for sky_model in ('21cm', 'GDSM', 'GSSM', 'PS'):
        if config_dict["sky_models"][sky_model]["include"]:
            if config_dict["sky_models"][sky_model]["image"] == '':
                errh.raise_error(ValueError,
                                 f"If including {sky_model} sky model, must "
                                 "specify a valid path for image")

            if not config_dict["sky_models"][sky_model]["create"]:
                im = pathlib.Path(config_dict["sky_models"][sky_model]["image"])
                if not im.exists():
                    err_msg = f"Using existing file, {im.__str__()}, as " \
                              f"{sky_model} sky model, but it doesn't exist"
                    errh.raise_error(FileNotFoundError,
                                     err_msg)


def load_configuration(toml_file: typing.Union[pathlib.Path, str]) -> dict:
    if not isinstance(toml_file, pathlib.Path):
        toml_file = pathlib.Path(toml_file)

    check_file_exists(toml_file)
    LOGGER.debug(f"{toml_file.resolve().__str__()} exists")

    config_dict = toml.load(toml_file)
    check_config_validity(config_dict)
    LOGGER.debug(f"{toml_file.resolve().__str__()} configuration is valid")

    return config_dict

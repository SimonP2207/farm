"""
All methods related to the loading of FARM configuration files
"""
import datetime
import random
from pathlib import Path
from typing import Union, Tuple
from dataclasses import dataclass

import numpy as np
import toml

import astropy.units as u
from astropy.coordinates import SkyCoord

from .. import LOGGER
from ..miscellaneous import error_handling as errh


def check_file_exists(filename: Path) -> bool:
    if not filename.exists():
        return False
    return True


def check_config_validity(config_dict: dict):
    """
    Check a dict loaded from a .toml configuration file has all required
    parameters

    Parameters
    ----------
    config_dict
        Python dict containing all required configuration parameters

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the configuration dict doesn't contain one or more required
        parameters
    TypeError
        If any setting value's type is incorrect
    FileNotFoundError
        If any images specified in the sky_models configuration section do not
        exist
    """
    structure = (('directories', 'root_name', str),
                 ('directories', 'telescope_model', str),
                 ('directories', 'output_dcy', str),
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
                 ('calibration', 'noise', 'include', bool),
                 ('calibration', 'noise', 'seed', int),
                 ('calibration', 'noise', 'sefd_frequencies_file', str),
                 ('calibration', 'noise', 'sefd_file', str),
                 ('calibration', 'TEC', 'include', bool),
                 ('calibration', 'gains', 'include', bool),
                 ('calibration', 'DD-effects', 'include', bool),
                 ('sky_models', '21cm', 'include', bool),
                 ('sky_models', '21cm', 'create', bool),
                 ('sky_models', '21cm', 'image', str),
                 ('sky_models', 'A-Team', 'include', bool),
                 ('sky_models', 'A-Team', 'demix_error', float),
                 ('sky_models', 'GDSM', 'include', bool),
                 ('sky_models', 'GDSM', 'create', bool),
                 ('sky_models', 'GDSM', 'image', str),
                 ('sky_models', 'GSSM', 'include', bool),
                 ('sky_models', 'GSSM', 'create', bool),
                 ('sky_models', 'GSSM', 'image', str),
                 ('sky_models', 'PS', 'include', bool),
                 ('sky_models', 'PS', 'create', bool),
                 ('sky_models', 'PS', 'image', str),
                 ('sky_models', 'PS', 'flux_inner', float),
                 ('sky_models', 'PS', 'flux_outer', float),
                 ('sky_models', 'TRECS', 'include', bool),
                 ('sky_models', 'TRECS', 'create', bool),
                 ('sky_models', 'TRECS', 'image', str),
                 ('sky_models', 'TRECS', 'flux_inner', float))

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
            # if config_dict["sky_models"][sky_model]["image"] == '':
            #     errh.raise_error(ValueError,
            #                      f"If including {sky_model} sky model, must "
            #                      "specify a valid path for image")

            if not config_dict["sky_models"][sky_model]["create"]:
                im = Path(config_dict["sky_models"][sky_model]["image"])
                if not im.exists():
                    err_msg = f"Using existing file, {im.__str__()}, as " \
                              f"{sky_model} sky model, but it doesn't exist"
                    errh.raise_error(FileNotFoundError, err_msg)


def load_configuration(toml_file: Union[Path, str]) -> dict:
    if not isinstance(toml_file, Path):
        toml_file = Path(toml_file)

    if not check_file_exists(toml_file):
        errh.raise_error(FileNotFoundError,
                         f"{str(toml_file.resolve())} doesn't exist")

    LOGGER.debug(f"{str(toml_file.resolve())} found")
    config_dict = toml.load(toml_file)
    check_config_validity(config_dict)
    LOGGER.debug(f"{str(toml_file.resolve())} configuration is valid")

    return config_dict


@dataclass
class Observation:
    time: datetime.datetime  # start time of observation
    duration: float  # s
    n_scan: int  #
    t_scan: float  # s

    @property
    def t_total(self):
        return self.n_scan * self.t_scan

@dataclass
class Field:
    _ra0: str
    _dec0: str
    _frame: str
    nx: int
    ny: int
    cdelt: float  # deg

    @property
    def coord0(self):
        return SkyCoord(self._ra0, self._dec0, frame=self._frame,
                        unit=(u.hourangle, u.deg))

    @property
    def fov(self):
        return self.nx * self.cdelt, self.ny * self.cdelt

    @property
    def area(self):
        return np.prod(self.fov)


@dataclass
class Correlator:
    freq_min: float  # minimum frequency [Hz]
    freq_max: float  # maximum frequency [Hz]
    n_chan: int  # number of evenly spaced channels from freq_min to freq_max
    chan_width: float  # channel width [Hz]
    t_int: float  # visibility integration time

    @property
    def frequencies(self):
        return np.linspace(self.freq_min, self.freq_max, self.n_chan)

    @property
    def freq_inc(self):
        return self.frequencies[1] - self.frequencies[0]


@dataclass
class Calibration:
    noise: bool
    tec: bool
    gains: bool
    dd_effects: bool
    noise_seed: int = random.randint(1, 1e6)
    sefd_freq_file: Union[str, Path] = ""
    sefd_file: Union[str, Path] = ""

    def __post_init__(self):
        self.sefd_freq_file = Path(self.sefd_freq_file)
        self.sefd_file = Path(self.sefd_file)
        self.sefd_rms_file = None

        if self.noise:
            if self.sefd_freq_file == "":
                errh.raise_error(ValueError,
                                 "sefd_freq_file not specified")

            if self.sefd_file == "":
                errh.raise_error(ValueError, "sefd_file not specified")

            for file in (self.sefd_file, self.sefd_freq_file):
                if not file.exists():
                    errh.raise_error(FileNotFoundError,
                                     f"{str(file)} doesn't exist")

                elif not file.is_file():
                    errh.raise_error(FileNotFoundError,
                                     f"{str(file)} is not a file")

    def create_sefd_rms_file(self, file_name: Path, *args, **kwargs):
        from ..calibration.noise import sefd_to_rms

        sefd_rms = sefd_to_rms(np.loadtxt(self.sefd_file), *args, **kwargs)
        np.savetxt(file_name, sefd_rms)
        self.sefd_rms_file = file_name


@dataclass
class Station:
    station_model: Path
    position: Tuple[float, float, float]

    def __post_init__(self):
        ants_position_file = self.station_model / 'layout.txt'
        xs, ys = np.loadtxt(ants_position_file).T

        self.ants = {}
        for number in range(len(xs)):
            self.ants[number] = xs[number], ys[number]

        self.n_ant = len(self.ants)


@dataclass
class Telescope:
    model: Path

    def __post_init__(self):
        self.lat, self.lon = np.loadtxt(self.model / 'position.txt')

        station_position_file = self.model / 'layout.txt'
        xs, ys, zs = np.loadtxt(station_position_file).T

        stations = []
        for f in self.model.iterdir():
            if f.is_dir() and 'station' in f.name:
                stations.append(f)
        stations.sort()

        self.stations = {}
        for s in stations:
            number = int(s.name.strip('station'))
            station_position = xs[number], ys[number], zs[number]
            self.stations[number] = Station(stations[number], station_position)

        self.n_stations = len(self.stations)


@dataclass
class SkyComponentConfiguration:
    create: bool
    image: Union[str, Path]
    flux_inner: Union[None, float] = None
    flux_outer: Union[None, float] = None
    demix_error: Union[None, float] = None


@dataclass
class SkyModelConfiguration:
    h21cm: Union[bool, SkyComponentConfiguration]
    ateam: Union[bool, SkyComponentConfiguration]
    gdsm: Union[bool, SkyComponentConfiguration]
    gssm: Union[bool, SkyComponentConfiguration]
    point_sources: Union[bool, SkyComponentConfiguration]
    trecs: Union[bool, SkyComponentConfiguration]


class FarmConfiguration:
    def __init__(self, configuration_file: Path):
        self.cfg = load_configuration(configuration_file)
        self.cfg_file = configuration_file

        self.root_name = Path(self.cfg["directories"]['root_name'])
        self.output_dcy = Path(self.cfg["directories"]['output_dcy'])

        tscp_model = Path(self.cfg["directories"]['telescope_model'])
        self.telescope = Telescope(tscp_model)

        cfg_observation = self.cfg["observation"]
        self.observation = Observation(cfg_observation["time"],
                                       cfg_observation["duration"],
                                       cfg_observation["n_scan"],
                                       cfg_observation["t_scan"])

        cfg_field = cfg_observation["field"]
        self.field = Field(cfg_field["ra0"], cfg_field["dec0"],
                           cfg_field["frame"],
                           cfg_field["nxpix"], cfg_field["nypix"],
                           cfg_field["cdelt"] / 3600.)

        cfg_correlator = cfg_observation["correlator"]
        self.correlator = Correlator(cfg_correlator["freq_min"],
                                     cfg_correlator["freq_max"],
                                     cfg_correlator["nchan"],
                                     cfg_correlator["chanwidth"],
                                     cfg_correlator["t_int"],)

        cfg_calibration = self.cfg["calibration"]
        self.calibration = Calibration(
            noise=cfg_calibration["noise"]["include"],
            tec=cfg_calibration["TEC"]["include"],
            gains=cfg_calibration["gains"]["include"],
            dd_effects=cfg_calibration["DD-effects"]["include"],
            noise_seed=cfg_calibration["noise"]["seed"],
            sefd_freq_file=cfg_calibration["noise"]["sefd_frequencies_file"],
            sefd_file=cfg_calibration["noise"]["sefd_file"]
        )

        self.calibration.create_sefd_rms_file(
            self.output_dcy / "sefd_rms.txt",
            len(np.loadtxt(self.telescope.model / 'layout.txt')),
            self.observation.t_total,
            self.correlator.chan_width
        )

        cfg_sky_models = self.cfg["sky_models"]
        h21cm, ateam, gdsm, gssm, point_sources, trecs = (False, ) * 6
        if cfg_sky_models["21cm"]["include"]:
            h21cm = SkyComponentConfiguration(
                cfg_sky_models["21cm"]["create"],
                cfg_sky_models["21cm"]["image"]
            )

        if cfg_sky_models["A-Team"]["include"]:
            ateam = SkyComponentConfiguration(
                create=False,
                image="",
                demix_error=cfg_sky_models["A-Team"]["demix_error"]
            )

        if cfg_sky_models["GDSM"]["include"]:
            gdsm = SkyComponentConfiguration(
                cfg_sky_models["GDSM"]["create"],
                cfg_sky_models["GDSM"]["image"]
            )

        if cfg_sky_models["GSSM"]["include"]:
            gssm = SkyComponentConfiguration(
                cfg_sky_models["GSSM"]["create"],
                cfg_sky_models["GSSM"]["image"]
            )

        if cfg_sky_models["PS"]["include"]:
            point_sources = SkyComponentConfiguration(
                cfg_sky_models["PS"]["create"],
                cfg_sky_models["PS"]["image"],
                cfg_sky_models["PS"]["flux_inner"],
                cfg_sky_models["PS"]["flux_outer"]
            )

        if cfg_sky_models["TRECS"]["include"]:
            trecs = SkyComponentConfiguration(
                cfg_sky_models["TRECS"]["create"],
                cfg_sky_models["TRECS"]["image"],
                cfg_sky_models["TRECS"]["flux_inner"],
            )

        self.sky_model = SkyModelConfiguration(
            h21cm=h21cm,
            ateam=ateam,
            gdsm=gdsm,
            gssm=gssm,
            point_sources=point_sources,
            trecs=trecs,
        )

        # TODO: SORT THIS BIT OUT. EACH TYPE OF OSKAR TASK NEEDS A SETTING FILE.
        #  DECIDE WHETHER TO WRITE AND USE AN INI FILE OR TO USE OSKAR'S
        #  PYTHON IMPLEMENTATION
        import oskar
        self.oskar_sim_interferometer_settings = oskar.SettingsTree(
            'oskar_sim_interferometer'
        )


"""
All methods related to the loading of FARM configuration files
"""
import datetime
import random
from pathlib import Path
from typing import Union, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import toml

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .. import LOGGER
from ..miscellaneous import error_handling as errh
from ..miscellaneous import decorators


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
    @dataclass
    class Param:
        """Class to handle a single configuration parameter"""
        keys: Tuple[str]
        param_type: type

        def return_val(self, dict_: dict) -> Union[Any, Exception]:
            """Return value from a dictionary corresponding to Param instance"""
            entry = config_dict.copy()

            try:
                for key in self.keys:
                    entry = entry[key]
                return entry

            except Exception as e:
                return e

    structure = (
        Param(('directories', 'root_name'), str),
        Param(('directories', 'telescope_model'), str),
        Param(('directories', 'output_dcy'), str),
        Param(('observation', 'time'), datetime.datetime),
        Param(('observation', 't_total'), int),
        Param(('observation', 'n_scan'), int),
        Param(('observation', 'min_gap_scan'), int),
        Param(('observation', 'min_elevation'), float),
        Param(('observation', 'field', 'ra0'), str),
        Param(('observation', 'field', 'dec0'), str),
        Param(('observation', 'field', 'frame'), str),
        Param(('observation', 'field', 'nxpix'), int),
        Param(('observation', 'field', 'nypix'), int),
        Param(('observation', 'field', 'cdelt'), float),
        Param(('observation', 'correlator', 'freq_min'), float),
        Param(('observation', 'correlator', 'freq_max'), float),
        Param(('observation', 'correlator', 'nchan'), int),
        Param(('observation', 'correlator', 'chanwidth'), float),
        Param(('calibration', 'noise', 'include'), bool),
        Param(('calibration', 'noise', 'seed'), int),
        Param(('calibration', 'noise', 'sefd_frequencies_file'), str),
        Param(('calibration', 'noise', 'sefd_file'), str),
        Param(('calibration', 'TEC', 'include'), bool),
        Param(('calibration', 'TEC', 'residual_error'), float),
        Param(('calibration', 'TEC', 'create'), bool),
        Param(('calibration', 'TEC', 'image'), list),
        Param(('calibration', 'gains', 'include'), bool),
        Param(('calibration', 'gains', 'phase_err'), float),
        Param(('calibration', 'gains', 'amp_err'), float),
        Param(('calibration', 'DD-effects', 'include'), bool),
        Param(('sky_models', '21cm', 'include'), bool),
        Param(('sky_models', '21cm', 'create'), bool),
        Param(('sky_models', '21cm', 'image'), str),
        Param(('sky_models', 'A-Team', 'include'), bool),
        Param(('sky_models', 'A-Team', 'demix_error'), float),
        Param(('sky_models', 'GDSM', 'include'), bool),
        Param(('sky_models', 'GDSM', 'create'), bool),
        Param(('sky_models', 'GDSM', 'image'), str),
        Param(('sky_models', 'GSSM', 'include'), bool),
        Param(('sky_models', 'GSSM', 'create'), bool),
        Param(('sky_models', 'GSSM', 'image'), str),
        Param(('sky_models', 'EG', 'include'), bool),
        Param(('sky_models', 'EG', 'flux_inner'), float),
        Param(('sky_models', 'EG', 'flux_outer'), float),
        Param(('sky_models', 'EG', 'flux_transition'), float),
        Param(('sky_models', 'EG', 'Known', 'include'), bool),
        Param(('sky_models', 'EG', 'Known', 'image'), str),
        Param(('sky_models', 'EG', 'TRECS', 'include'), bool),
        Param(('sky_models', 'EG', 'TRECS', 'create'), bool),
        Param(('sky_models', 'EG', 'TRECS', 'image'), str)
    )

    for param in structure:
        return_val = param.return_val(config_dict)
        if isinstance(return_val, KeyError):
            errh.raise_error(KeyError,
                             f"{'.'.join(param.keys)} not present in config")
        elif not isinstance(return_val, param.param_type):
            errh.raise_error(TypeError,
                             f"{'.'.join(param.keys)} incorrect type. "
                             f"{param.param_type} required but "
                             f"{type(return_val)} provided")

    existing_image_params = (
        Param(('directories', 'telescope_model'), str),
        Param(('directories', 'output_dcy'), str),
        Param(('observation', 'time'), datetime.datetime),
        Param(('observation', 't_total'), int),
        Param(('observation', 'n_scan'), int),
        Param(('observation', 'min_gap_scan'), int),
        Param(('observation', 'min_elevation'), float),
        Param(('observation', 'field', 'ra0'), str),
        Param(('observation', 'field', 'dec0'), str),
        Param(('observation', 'field', 'frame'), str),
        Param(('observation', 'field', 'nxpix'), int),
        Param(('observation', 'field', 'nypix'), int),
        Param(('observation', 'field', 'cdelt'), float),
        Param(('observation', 'correlator', 'freq_min'), float),
        Param(('observation', 'correlator', 'freq_max'), float),
        Param(('observation', 'correlator', 'nchan'), int),
        Param(('observation', 'correlator', 'chanwidth'), float),
        Param(('calibration', 'noise', 'include'), bool),
        Param(('calibration', 'noise', 'seed'), int),
        Param(('calibration', 'noise', 'sefd_frequencies_file'), str),
        Param(('calibration', 'noise', 'sefd_file'), str),
        Param(('calibration', 'TEC', 'include'), bool),
        Param(('calibration', 'TEC', 'residual_error'), float),
        Param(('calibration', 'TEC', 'create'), bool),
        Param(('calibration', 'TEC', 'image'), list),
        Param(('calibration', 'gains', 'include'), bool),
        Param(('calibration', 'gains', 'phase_err'), float),
        Param(('calibration', 'gains', 'amp_err'), float),
        Param(('calibration', 'DD-effects', 'include'), bool),
        Param(('sky_models', '21cm', 'include'), bool),
        Param(('sky_models', '21cm', 'create'), bool),
        Param(('sky_models', '21cm', 'image'), str),
        Param(('sky_models', 'A-Team', 'include'), bool),
        Param(('sky_models', 'A-Team', 'demix_error'), float),
        Param(('sky_models', 'GDSM', 'include'), bool),
        Param(('sky_models', 'GDSM', 'create'), bool),
        Param(('sky_models', 'GDSM', 'image'), str),
        Param(('sky_models', 'GSSM', 'include'), bool),
        Param(('sky_models', 'GSSM', 'create'), bool),
        Param(('sky_models', 'GSSM', 'image'), str),
        Param(('sky_models', 'EG', 'include'), bool),
        Param(('sky_models', 'EG', 'flux_inner'), float),
        Param(('sky_models', 'EG', 'flux_outer'), float),
        Param(('sky_models', 'EG', 'flux_transition'), float),
        Param(('sky_models', 'EG', 'Known', 'include'), bool),
        Param(('sky_models', 'EG', 'Known', 'image'), str),
        Param(('sky_models', 'EG', 'TRECS', 'include'), bool),
        Param(('sky_models', 'EG', 'TRECS', 'image'), str)
    )

    # Check files/directories exist that must
    must_exist_files = []
    if config_dict['calibration']['noise']['include']:
        must_exist_files += (
            config_dict['calibration']['noise']['sefd_frequencies_file'],
            config_dict['calibration']['noise']['sefd_file']
        )

    if config_dict['calibration']['TEC']['include']:
        if not config_dict['calibration']['TEC']['create']:
            must_exist_files += config_dict['calibration']['TEC']['image']

    if config_dict['sky_models']['21cm']['include']:
        if not config_dict['sky_models']['21cm']['create']:
            must_exist_files.append(config_dict['sky_models']['21cm']['image'])

    if config_dict['sky_models']['GDSM']['include']:
        if not config_dict['sky_models']['GDSM']['create']:
            must_exist_files.append(config_dict['sky_models']['GDSM']['image'])

    if config_dict['sky_models']['GSSM']['include']:
        if not config_dict['sky_models']['GSSM']['create']:
            must_exist_files.append(config_dict['sky_models']['GSSM']['image'])

    if config_dict['sky_models']['EG']['Known']['include']:
        must_exist_files.append(config_dict['sky_models']['EG']['Known']['image'])

    if config_dict['sky_models']['EG']['Known']['include']:
        must_exist_files.append(config_dict['sky_models']['EG']['Known']['image'])

    for file in must_exist_files:
        if not Path(file).exists() and file != "":
            err_msg = f"{file} doesn't exist"
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
    time: Time  # start time of observation
    t_total: int  # s
    n_scan: int  # s
    min_gap_scan: int  # s
    min_elevation: int


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
class TEC:
    create: bool
    image: List[Path]
    err: float = field(default=0.0)

    def __post_init__(self):
        if not self.create:
            if not self.image:
                raise ValueError(f"If not creating TEC screen, image(s) must "
                                 f"be provided")

            for idx, im in enumerate(self.image):
                if isinstance(im, str):
                    if im == '':
                        raise ValueError("Empty string is not a valid image")
                    self.image[idx] = Path(im)

                if not self.image[idx].exists():
                    raise FileNotFoundError(f"TEC image, {self.image[idx]}, not"
                                            f" found")
        else:
            self.image = []

        if not 0.0 < self.err < 1.0:
            raise ValueError(f"Invalid TEC error value given ({self.err:.3f})")


@dataclass
class Noise:
    seed: int
    sefd_freq_file: Union[str, Path]
    sefd_file: Union[str, Path]
    sefd_rms_file: Union[Path, None] = field(default=None, init=False)

    def __post_init__(self):
        if isinstance(self.sefd_freq_file, str):
            self.sefd_freq_file = Path(self.sefd_freq_file)

        if isinstance(self.sefd_file, str):
            self.sefd_file = Path(self.sefd_file)

        if not self.sefd_freq_file.exists():
            raise FileNotFoundError("SEFD frequencies data file, "
                                    f"{self.sefd_freq_file}, not found")

        if not self.sefd_file.exists():
            raise FileNotFoundError(f"SEFD data file, {self.sefd_file}, not "
                                    f"found")

        if not self._valid_sefd_setup:
            raise ValueError("SEFD data/SEFD frequencies files not compatible")

    @property
    def _valid_sefd_setup(self) -> bool:
        freqs = np.loadtxt(self.sefd_freq_file)
        sefds = np.loadtxt(self.sefd_file)

        if len(freqs) != len(sefds):
            return False

        return True

    def create_sefd_rms_file(self, file_name: Path, *args, **kwargs):
        from ..calibration.noise import sefd_to_rms

        sefd_rms = sefd_to_rms(np.loadtxt(self.sefd_file), *args, **kwargs)
        np.savetxt(file_name, sefd_rms)

        self.sefd_rms_file = file_name


@dataclass
class Gains:
    amp_err: float = field(default=0.0)
    phase_err: float = field(default=0.0)

    def __post_init__(self):
        if self.amp_err < 0.0 or self.amp_err > 100.0:
            raise ValueError(f"Invalid residual amplitude error value given "
                             f"for gains ({self.err:.2f}). Should be "
                             f"0 < err <= 100")
        if self.phase_err < 0.0:
            raise ValueError(f"Invalid residual phase error value given "
                             f"for gains ({self.err:.2f}). Should be >= 0")


@dataclass
class DDEffects:
    pass


@dataclass
class Calibration:
    noise: Union[bool, Noise]
    tec: Union[bool, TEC]
    gains: Union[bool, Gains]
    dd_effects: Union[bool, DDEffects]


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
        from astropy.coordinates import EarthLocation

        self.lon, self.lat = np.loadtxt(self.model / 'position.txt')

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
        self.location = EarthLocation(lat=self.lat * u.deg,
                                      lon=self.lon * u.deg)


@dataclass
class SkyComponentConfiguration:
    create: bool
    image: Union[str, Path]
    flux_range: Tuple[float, float] = (0.0, 1e30)
    flux_inner: Union[None, float] = None
    flux_outer: Union[None, float] = None
    demix_error: Union[None, float] = None


@dataclass
class SkyModelConfiguration:
    h21cm: Union[bool, SkyComponentConfiguration]
    ateam: Union[bool, SkyComponentConfiguration]
    gdsm: Union[bool, SkyComponentConfiguration]
    gssm: Union[bool, SkyComponentConfiguration]
    extragal_known: Union[bool, SkyComponentConfiguration]
    extragal_trecs: Union[bool, SkyComponentConfiguration]


class FarmConfiguration:
    @decorators.suppress_warnings("astropy", "erfa")
    def __init__(self, configuration_file: Path):
        self.cfg = load_configuration(configuration_file)
        self.cfg_file = configuration_file

        # Directories setup
        self.output_dcy = Path(self.cfg["directories"]['output_dcy'])
        self.root_name = self.output_dcy / self.cfg["directories"]['root_name']

        # Telescope model setup
        tscp_model = Path(self.cfg["directories"]['telescope_model'])
        self.telescope = Telescope(tscp_model)

        # Observational details setup
        cfg_observation = self.cfg["observation"]
        # Creation of this Time instance can lead to warnings due to the date
        # being too far in the future
        t0 = Time(self.cfg["observation"]["time"].strftime("%Y-%m-%d %H:%M:%S"),
                  scale='utc', location=(f'{self.telescope.lon:.5f}d',
                                         f'{self.telescope.lat:.5f}d'))

        self.observation = Observation(t0,
                                       cfg_observation["t_total"],
                                       cfg_observation["n_scan"],
                                       cfg_observation["min_gap_scan"],
                                       cfg_observation["min_elevation"])

        # Observing field setup
        cfg_field = cfg_observation["field"]
        self.field = Field(cfg_field["ra0"], cfg_field["dec0"],
                           cfg_field["frame"],
                           cfg_field["nxpix"], cfg_field["nypix"],
                           cfg_field["cdelt"] / 3600.)

        # Correlator setup
        cfg_correlator = cfg_observation["correlator"]
        self.correlator = Correlator(cfg_correlator["freq_min"],
                                     cfg_correlator["freq_max"],
                                     cfg_correlator["nchan"],
                                     cfg_correlator["chanwidth"],
                                     cfg_correlator["t_int"],)

        # Calibration setup
        noise, tec, gains, dd_effects = False, False, False, False
        cfg_calibration = self.cfg["calibration"]

        if cfg_calibration["noise"]["include"]:
            noise = Noise(
                seed=cfg_calibration["noise"]["seed"],
                sefd_freq_file=cfg_calibration["noise"]["sefd_frequencies_file"],
                sefd_file=cfg_calibration["noise"]["sefd_file"]
            )
            noise.create_sefd_rms_file(
                self.output_dcy / "sefd_rms.txt",
                len(np.loadtxt(self.telescope.model / 'layout.txt')),
                self.observation.t_total,
                self.correlator.chan_width
            )

        if cfg_calibration["TEC"]["include"]:
            tec = TEC(create=cfg_calibration["TEC"]["create"],
                      image=cfg_calibration["TEC"]["image"],
                      err=cfg_calibration["TEC"]["residual_error"])

        if cfg_calibration["gains"]["include"]:
            gains = Gains(amp_err=cfg_calibration["gains"]["amp_err"],
                          phase_err=cfg_calibration["gains"]["phase_err"],)

        if cfg_calibration["DD-effects"]['include']:
            dd_effects = DDEffects()

        self.calibration = Calibration(noise=noise,
                                       tec=tec,
                                       gains=gains,
                                       dd_effects=dd_effects)

        # Sky model setup
        cfg_sky_models = self.cfg["sky_models"]
        h21cm, ateam, gdsm, gssm, known_sources, trecs = (False, ) * 6
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

        if cfg_sky_models["EG"]["include"]:
            if cfg_sky_models["EG"]["Known"]["include"]:
                known_sources = SkyComponentConfiguration(
                    False,
                    cfg_sky_models["EG"]["Known"]["image"],
                    flux_range=(cfg_sky_models["EG"]["flux_transition"], 1e30),
                    flux_inner=cfg_sky_models["EG"]["flux_inner"],
                    flux_outer=cfg_sky_models["EG"]["flux_outer"]
                )

            if cfg_sky_models["EG"]["TRECS"]["include"]:
                trecs = SkyComponentConfiguration(
                    cfg_sky_models["EG"]["TRECS"]["create"],
                    cfg_sky_models["EG"]["TRECS"]["image"],
                    flux_range=(0., cfg_sky_models["EG"]["flux_transition"]),
                    flux_inner=cfg_sky_models["EG"]["flux_inner"],
                    flux_outer=cfg_sky_models["EG"]["flux_outer"]
                )

        self.sky_model = SkyModelConfiguration(
            h21cm=h21cm,
            ateam=ateam,
            gdsm=gdsm,
            gssm=gssm,
            extragal_known=known_sources,
            extragal_trecs=trecs,
        )

        self.sky_model.image = (self.output_dcy /
                                f'{self.root_name}_sky_model.fits')
        self.oskar_sky_model_file = self.output_dcy / 'oskar_sky_sources.data'

        # TODO: SORT THIS BIT OUT. EACH TYPE OF OSKAR TASK NEEDS A SETTING FILE.
        #  DECIDE WHETHER TO WRITE AND USE AN INI FILE OR TO USE OSKAR'S
        #  PYTHON IMPLEMENTATION
        import oskar
        self.oskar_sim_interferometer_settings = oskar.SettingsTree(
            'oskar_sim_interferometer'
        )

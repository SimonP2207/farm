"""
All methods related to the loading of FARM configuration files
"""
import datetime
from pathlib import Path
from typing import Union, List, Tuple, Any, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import toml

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import EarthLocation

from .. import LOGGER
from ..miscellaneous import error_handling as errh
from ..miscellaneous import decorators


# TODO: Write check_tec_image_compatibility method
def check_tec_image_compatibility(farm_cfg: 'FarmConfiguration',
                                  tec_images: List[Path]
                                  ) -> Tuple[bool, str]:
    """
    Checks whether a list of TEC fits files is compatible with the observation
    specified within a FarmConfiguration instance

    Parameters
    ----------
    farm_cfg
        FarmConfiguration instance to parse information from
    tec_images
        List of paths to TEC-screen fits-files

    Returns
    -------
    Tuple of (bool, str) whereby the bool indicates compatibility and the str is
    contains the reason for incompatibility if False (empty string if True)
    """
    return True, ""


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
            entry = dict_.copy()

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
        Param(('sky_models', 'Galactic', 'LargeScale', 'include'), bool),
        Param(('sky_models', 'Galactic', 'LargeScale', 'create'), bool),
        Param(('sky_models', 'Galactic', 'LargeScale', 'image'), str),
        Param(('sky_models', 'Galactic', 'SmallScale', 'include'), bool),
        Param(('sky_models', 'Galactic', 'SmallScale', 'create'), bool),
        Param(('sky_models', 'Galactic', 'SmallScale', 'image'), str),
        Param(('sky_models', 'EG', 'include'), bool),
        Param(('sky_models', 'EG', 'image'), str),
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

    if config_dict['sky_models']['Galactic']['LargeScale']['include']:
        if not config_dict['sky_models']['Galactic']['LargeScale']['create']:
            must_exist_files.append(
                config_dict['sky_models']['Galactic']['LargeScale']['image'])

    if config_dict['sky_models']['Galactic']['SmallScale']['include']:
        if not config_dict['sky_models']['Galactic']['SmallScale']['create']:
            must_exist_files.append(
                config_dict['sky_models']['Galactic']['SmallScale']['image'])

    if config_dict['sky_models']['EG']['Known']['include']:
        must_exist_files.append(
            config_dict['sky_models']['EG']['Known']['image'])

    if config_dict['sky_models']['EG']['Known']['include']:
        must_exist_files.append(
            config_dict['sky_models']['EG']['Known']['image'])

    for file in must_exist_files:
        if not Path(file).exists() and file != "":
            err_msg = f"{file} doesn't exist"
            errh.raise_error(FileNotFoundError, err_msg)


def load_configuration(toml_file: Union[Path, str]) -> dict:
    """Load a .toml configuration file and return it as a dict"""
    if not isinstance(toml_file, Path):
        toml_file = Path(toml_file)

    if not toml_file.exists():
        errh.raise_error(FileNotFoundError,
                         f"{str(toml_file.resolve())} doesn't exist")

    LOGGER.debug(f"{str(toml_file.resolve())} found")
    config_dict = toml.load(toml_file)
    check_config_validity(config_dict)
    LOGGER.debug(f"{str(toml_file.resolve())} configuration is valid")

    return config_dict


@dataclass
class Observation:
    """
    Class containing all information and methods pertaining to a full synthetic
    observation of variable numbers of scans
    """
    t_start: Time  # start time of observation
    t_total: int  # s
    n_scan: int  # s
    min_gap_scan: int  # s
    min_elevation: int  # deg

    def scan_times(
            self, coord0: SkyCoord, location: EarthLocation,
            partial_scans_allowed: bool = False
    ) -> Tuple[Tuple[Time, Time], ...]:
        """
        Calculates scan_times for the Observation for a particular pointing
        centre, from a specific location

        Parameters
        ----------
        coord0
            Celestial coordinate to observe
        location
            Earth location from which to observe
        partial_scans_allowed
            Whether to allow scans to be broken across minimum elevation
            boundaries. Default is False

        Returns
        -------
        Tuple containing two-tuples of (astropy.time.Time, astropy.time.Time)
        representing scan start/end times
        """
        from ..physics import astronomy as ast

        # Compute scan times
        scans = ast.scan_times(
            self.t_start, coord0,
            location, self.n_scan,
            self.t_total, self.min_elevation,
            self.min_gap_scan, partial_scans_allowed=partial_scans_allowed
        )

        return tuple(scans)


@dataclass
class Field:
    """Class for holding observational field information"""
    _ra0: str
    _dec0: str
    _frame: str
    nx: int
    ny: int
    cdelt: float  # deg

    @property
    def coord0(self):
        """Pointing centre as a SkyCoord instance"""
        return SkyCoord(self._ra0, self._dec0, frame=self._frame,
                        unit=(u.hourangle, u.deg))

    @property
    def fov(self):
        """Field of view in x and y as a 2-tuple in deg"""
        return self.nx * self.cdelt, self.ny * self.cdelt

    @property
    def area(self):
        """Total area of field of view in deg^2"""
        return np.prod(self.fov)


@dataclass
class Correlator:
    """Class for handling all information pertaining to a Correlator setup"""
    freq_min: float  # minimum frequency [Hz]
    freq_max: float  # maximum frequency [Hz]
    n_chan: int  # number of evenly spaced channels from freq_min to freq_max
    chan_width: float  # channel width [Hz]
    t_int: float  # visibility integration time

    @property
    def frequencies(self):
        """All channel frequencies [Hz]"""
        return np.linspace(self.freq_min, self.freq_max, self.n_chan)

    @property
    def freq_inc(self):
        """Frequency gap between neighbouring channels"""
        return self.frequencies[1] - self.frequencies[0]


@dataclass
class TEC:
    """Class for handling all TEC-screen information"""
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
    """
    Class for handling informaiton needed for generation of noise within the
    synthetic observation
    """
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
        """Create file with System Equivalent Flux Density rms noises"""
        from ..calibration.noise import sefd_to_rms

        sefd_rms = sefd_to_rms(np.loadtxt(self.sefd_file), *args, **kwargs)
        np.savetxt(file_name, sefd_rms)

        self.sefd_rms_file = file_name


@dataclass
class Gains:
    """Class for handling implemented gain error information"""
    amp_err: float = field(default=0.0)
    phase_err: float = field(default=0.0)

    def __post_init__(self):
        if self.amp_err < 0.0 or self.amp_err > 100.0:
            raise ValueError(f"Invalid residual amplitude error value given "
                             f"for gains ({self.amp_err:.2f}). Should be "
                             f"0 < err <= 100")
        if self.phase_err < 0.0:
            raise ValueError(f"Invalid residual phase error value given "
                             f"for gains ({self.phase_err:.2f}). "
                             f"Should be >= 0")


@dataclass
class DDEffects:
    """Future class for handling direction-dependent errors and effects"""
    pass


@dataclass
class Calibration:
    """
    Composite dataclass containing of all calibration related class instances
    for Noise, TEC, Gains and DDEffects
    """
    noise: Union[bool, Noise]
    tec: Union[bool, TEC]
    gains: Union[bool, Gains]
    dd_effects: Union[bool, DDEffects]


@dataclass
class Station:
    """
    Dataclass for handling individual station information in an interferometric
    array
    """
    station_model: Path
    position: Tuple[float, float, float]
    ants: Dict = field(init=False)
    n_ant: int = field(init=False)

    def __post_init__(self):
        ants_position_file = self.station_model / 'layout.txt'
        xs, ys = np.loadtxt(ants_position_file).T

        self.ants = {}
        for number in range(len(xs)):
            self.ants[number] = xs[number], ys[number]

        self.n_ant = len(self.ants)


@dataclass
class Telescope:
    """Class containing all information related to an interferometer"""
    model: Path
    ref_ant: Optional[int] = field(default=None)
    lon: float = field(init=False)
    lat: float = field(init=False)
    stations: Dict[int, Station] = field(init=False)
    n_stations: int = field(init=False)
    location: EarthLocation = field(init=False)
    centre: npt.NDArray[float] = field(init=False)
    _baseline_lengths: Union[None, Dict[Tuple[int, int], float]] = field(init=0)

    def __post_init__(self):
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
            station_position = np.array([xs[number], ys[number], zs[number]])
            self.stations[number] = Station(stations[number], station_position)

        self.n_stations = len(self.stations)
        self.location = EarthLocation(lat=self.lat * u.deg,
                                      lon=self.lon * u.deg)
        self.centre = np.mean([v.position for k, v in self.stations.items()],
                              axis=0)

        # If no reference antenna is given, assign the central-most antenna in
        # the array as the reference antenna
        if self.ref_ant is None:
            station_dists = {}
            for k, s in self.stations.items():
                station_dists[k] = self.dist_to_centre(s.position)

            min_dist = 1e30
            for k, v in station_dists.items():
                if v < min_dist:
                    min_dist = v
                    self.ref_ant = k

        self._baseline_lengths = None

    @property
    def baseline_lengths(self) -> Dict[Tuple[int, int], float]:
        """
        Baseline lengths of all as a dict whose keys are 2-tuples of each
        baseline's antennae numbers and values are separations between those
        two antennae
        """
        if self._baseline_lengths is None:
            from itertools import combinations

            ant_pairs = combinations(self.stations.keys(), 2)
            self._baseline_lengths = {}
            for ant_pair in ant_pairs:
                positions = [self.stations[_].position for _ in ant_pair]
                baseline_length = self._dist_between_points(*positions)
                self._baseline_lengths[ant_pair] = baseline_length

        return self._baseline_lengths

    @staticmethod
    def _dist_between_points(pos1: npt.NDArray[float],
                             pos2: npt.NDArray[float]) -> float:
        """Dist [m] between two points"""
        return np.sqrt(np.sum((pos2 - pos1) ** 2.))

    def dist_to_centre(self, pos: npt.NDArray[float]) -> float:
        """Dist [m] to the array's geometrical centre"""
        return self._dist_between_points(pos, self.centre)

    def dist_to_refant(self, pos: npt.NDArray[float]) -> float:
        """Distance of a position to the reference antenna"""
        ref_station = self.stations[self.ref_ant]
        return self._dist_between_points(pos, ref_station.position)


# @dataclass
# class SkyComponentConfiguration:
#     create: bool
#     image: Union[str, Path]
#     flux_range: Tuple[float, float] = (0.0, 1e30)
#     flux_inner: Union[None, float] = None
#     flux_outer: Union[None, float] = None
#     demix_error: Union[None, float] = None

@dataclass
class SkyComponentConfiguration:
    """Parent class handling SkyComponent inclusion/loading"""
    include: bool
    image: Union[str, Path]

    def __post_init__(self):
        if isinstance(self.image, str) and self.image != "":
            self.image = Path(self.image)


@dataclass
class EoR21cmConfiguration(SkyComponentConfiguration):
    """Class handling epoch of reionisation hydrogen 21cm signal inclusion"""
    create: bool


@dataclass
class ATeamConfiguration(SkyComponentConfiguration):
    """Class handling A-Team inclusion and demixing error"""
    demix_error: float


@dataclass
class ExtragalacticComponentConfiguration(SkyComponentConfiguration):
    """Class handling extragalactic source inclusion/creation"""
    create: bool
    flux_inner: float
    flux_outer: float
    flux_transition: float


@dataclass
class ExtragalacticConfiguration(SkyComponentConfiguration):
    """
    Composite class composed of real and artificial extragalactic source
    inclusion as separate instances of ExtragalacticComponentConfiguration
    instances
    """
    real_component: Union[ExtragalacticComponentConfiguration, None]
    artifical_component: Union[ExtragalacticComponentConfiguration, None]


@dataclass
class GalacticComponentConfiguration(SkyComponentConfiguration):
    """Class handling inclusion information for the Galactic component"""
    create: bool


@dataclass
class GalacticConfiguration(SkyComponentConfiguration):
    """
    Composite class composed of Galactic small and large scale source
    inclusion as separate instances of GalacticComponentConfiguration
    instances
    """
    large_scale_component: Union[GalacticComponentConfiguration, None]
    small_scale_component: Union[GalacticComponentConfiguration, None]


@dataclass
class SkyModelConfiguration:
    """
    Composite class representing full SkyModel inclusion information comprised
    of H-21cm, A-Team, Galactic and Extragalactic component instances
    """
    h21cm: Union[bool, EoR21cmConfiguration]
    ateam: Union[bool, ATeamConfiguration]
    galactic: Union[bool, GalacticConfiguration]
    extragalactic: Union[bool, ExtragalacticConfiguration]


class FarmConfiguration:
    """
    Class to handle the configuration of running pipelines utilising the farm
    library
    """
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
                                     cfg_correlator["t_int"], )

        # Calibration setup
        noise, tec, gains, dd_effects = False, False, False, False
        cfg_calibration = self.cfg["calibration"]

        if not self.output_dcy.exists():
            self.output_dcy.mkdir(exist_ok=True)

        if cfg_calibration["noise"]["include"]:
            noise = Noise(
                seed=cfg_calibration["noise"]["seed"],
                sefd_freq_file=cfg_calibration["noise"][
                    "sefd_frequencies_file"],
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
                          phase_err=cfg_calibration["gains"]["phase_err"], )

        if cfg_calibration["DD-effects"]['include']:
            dd_effects = DDEffects()

        self.calibration = Calibration(noise=noise,
                                       tec=tec,
                                       gains=gains,
                                       dd_effects=dd_effects)

        # Instantiate all instances of configuration classes for all sky model
        # components
        cfg_sky_models = self.cfg["sky_models"]
        h21cm = EoR21cmConfiguration(
            include=cfg_sky_models["21cm"]["include"],
            image=cfg_sky_models["21cm"]["image"],
            create=cfg_sky_models["21cm"]["create"]
        )
        ateam = ATeamConfiguration(
            include=cfg_sky_models["A-Team"]["include"],
            image="",
            demix_error=cfg_sky_models["A-Team"]["demix_error"]
        )

        gdsm = GalacticComponentConfiguration(
            include=cfg_sky_models['Galactic']['LargeScale']["include"],
            image=cfg_sky_models['Galactic']['LargeScale']["image"],
            create=cfg_sky_models['Galactic']['LargeScale']["create"]
        )

        gssm = GalacticComponentConfiguration(
            include=cfg_sky_models['Galactic']['SmallScale']["include"],
            image=cfg_sky_models['Galactic']['SmallScale']["image"],
            create=cfg_sky_models['Galactic']['SmallScale']["create"]
        )

        galactic = GalacticConfiguration(
            include=cfg_sky_models['Galactic']["include"],
            image=cfg_sky_models['Galactic']["image"],
            large_scale_component=gdsm if gdsm else None,
            small_scale_component=gssm if gssm else None
        )

        known_sources = ExtragalacticComponentConfiguration(
            include=cfg_sky_models["EG"]["Known"]["include"],
            image=cfg_sky_models["EG"]["Known"]["image"],
            create=False,
            flux_inner=cfg_sky_models["EG"]["flux_inner"],
            flux_outer=cfg_sky_models["EG"]["flux_outer"],
            flux_transition=cfg_sky_models["EG"]["flux_transition"]
        )

        trecs = ExtragalacticComponentConfiguration(
            include=cfg_sky_models["EG"]["TRECS"]["include"],
            image=cfg_sky_models["EG"]["TRECS"]["image"],
            create=cfg_sky_models["EG"]["TRECS"]["create"],
            flux_inner=1e30,
            flux_outer=-1e30,
            flux_transition=cfg_sky_models["EG"]["flux_transition"]
        )
        extragalactic = ExtragalacticConfiguration(
            include=cfg_sky_models["EG"]["include"],
            image=cfg_sky_models["EG"]["image"],
            real_component=known_sources,
            artifical_component=trecs
        )

        self.sky_model = SkyModelConfiguration(
            h21cm=h21cm,
            ateam=ateam,
            galactic=galactic,
            extragalactic=extragalactic,
        )

        # TODO: Should the image attribute be assigned in the
        #  SkyModelConfiguration class?
        self.sky_model.image = (self.output_dcy /
                                f'{self.root_name}_sky_model.fits')
        self.oskar_sky_model_file = self.output_dcy / 'oskar_sky_sources.data'

        self.sbeam_ini = self.setup_sim_beam_pattern_ini()
        self.sinterferometer_ini = self.setup_sim_interferometer_ini()

    def setup_sim_interferometer_ini(self):
        """
        Sets up .ini file for oskar's sim_interferometer task from configuration
        parameters
        """
        from ..software.oskar import set_oskar_sim_interferometer

        ini_file = Path(f"{self.root_name}_sim_interferometer.ini")

        LOGGER.info("Setting up interferometer .ini files")
        with open(ini_file, 'wt') as f:
            set_oskar_sim_interferometer(f, 'simulator/double_precision',
                                         'TRUE')
            set_oskar_sim_interferometer(f, 'simulator/use_gpus', 'FALSE')
            set_oskar_sim_interferometer(f, 'simulator/max_sources_per_chunk',
                                         '4096')

            set_oskar_sim_interferometer(f, 'observation/phase_centre_ra_deg',
                                         self.field.coord0.ra.deg)
            set_oskar_sim_interferometer(f, 'observation/phase_centre_dec_deg',
                                         self.field.coord0.dec.deg)
            set_oskar_sim_interferometer(f, 'observation/start_frequency_hz',
                                         self.correlator.freq_min)
            set_oskar_sim_interferometer(f, 'observation/num_channels',
                                         self.correlator.n_chan)
            set_oskar_sim_interferometer(f, 'observation/frequency_inc_hz',
                                         self.correlator.freq_inc)
            set_oskar_sim_interferometer(f, 'telescope/input_directory',
                                         self.telescope.model)
            set_oskar_sim_interferometer(
                f, 'telescope/allow_station_beam_duplication', 'TRUE'
            )
            set_oskar_sim_interferometer(f, 'telescope/pol_mode', 'Scalar')

            # Add in ionospheric screen model
            set_oskar_sim_interferometer(f, 'telescope/ionosphere_screen_type',
                                         'External')
            set_oskar_sim_interferometer(f,
                                         'interferometer/channel_bandwidth_hz',
                                         self.correlator.chan_width)
            set_oskar_sim_interferometer(f, 'interferometer/time_average_sec',
                                         self.correlator.t_int)
            set_oskar_sim_interferometer(f,
                                         'interferometer/ignore_w_components',
                                         'FALSE')

            # Add in Telescope noise model via files where rms has been tuned
            set_oskar_sim_interferometer(
                f, 'interferometer/noise/enable',
                True if self.calibration.noise else False
            )
            set_oskar_sim_interferometer(f, 'interferometer/noise/seed',
                                         self.calibration.noise.seed)
            set_oskar_sim_interferometer(f, 'interferometer/noise/freq', 'Data')
            set_oskar_sim_interferometer(f, 'interferometer/noise/freq/file',
                                         self.calibration.noise.sefd_freq_file)
            set_oskar_sim_interferometer(f, 'interferometer/noise/rms', 'Data')
            set_oskar_sim_interferometer(f, 'interferometer/noise/rms/file',
                                         self.calibration.noise.sefd_rms_file)
            # set_oskar_sim_interferometer(f, 'sky/fits_image/file',
            #                              self.sky_model.image)
            # set_oskar_sim_interferometer(f, 'sky/fits_image/default_map_units',
            #                              'K')

        return ini_file

    def setup_sim_beam_pattern_ini(self):
        """
        Sets up .ini file for oskar's sim_beam_pattern task from configuration
        parameters
        """
        from ..software.oskar import set_oskar_sim_beam_pattern

        ini_file = Path(f'{self.root_name}_sim_beam.ini')

        LOGGER.info(f"Setting up oskar's sim_beam_pattern .ini file, "
                    f"{ini_file}")
        with open(ini_file, 'wt') as f:
            set_oskar_sim_beam_pattern(f, "simulator/double_precision", False)
            set_oskar_sim_beam_pattern(f, "observation/phase_centre_ra_deg",
                                       self.field.coord0.ra.deg)
            set_oskar_sim_beam_pattern(f, "observation/phase_centre_dec_deg",
                                       self.field.coord0.dec.deg)
            set_oskar_sim_beam_pattern(f, "observation/start_frequency_hz",
                                       self.correlator.freq_min)
            set_oskar_sim_beam_pattern(f, "observation/num_channels",
                                       self.correlator.n_chan)
            set_oskar_sim_beam_pattern(f, "observation/frequency_inc_hz",
                                       self.correlator.freq_inc)
            set_oskar_sim_beam_pattern(f, "observation/num_time_steps", 1)
            set_oskar_sim_beam_pattern(f, "telescope/input_directory",
                                       self.telescope.model)
            set_oskar_sim_beam_pattern(f, "telescope/pol_mode",
                                       "Scalar")
            # 1.02 factor here ensures no nasty edge effects
            set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/fov_deg",
                                       self.field.fov[0] * 1.02)
            set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/size",
                                       self.field.nx)
            set_oskar_sim_beam_pattern(
                f, "beam_pattern/station_outputs/fits_image/auto_power", True
            )

        return ini_file

    def set_oskar_sim_interferometer(self, key, value):
        """Wrapper around farm.software.oskar.set_oskar_sim_interferometer"""
        from ..software.oskar import set_oskar_sim_interferometer

        with open(self.sinterferometer_ini, 'at') as f:
            set_oskar_sim_interferometer(f, key, value)

    def set_oskar_sim_beam_pattern(self, key, value):
        """Wrapper around farm.software.oskar.set_oskar_sim_beam_pattern"""
        from ..software.oskar import set_oskar_sim_beam_pattern

        with open(self.sbeam_ini, 'at') as f:
            set_oskar_sim_beam_pattern(f, key, value)

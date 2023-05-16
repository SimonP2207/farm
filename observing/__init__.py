from .instruments import Correlator, Subband, Telescope
from .observation import generate_scans
from .observation import Field, Observation, Scan
# import datetime
# import pathlib
# import shutil
# from typing import Iterable, List, Optional, Tuple, Union
# from dataclasses import dataclass
# import numpy as np
# from astropy.coordinates import EarthLocation, SkyCoord
# from astropy.time import Time
# from astropy.io import fits
#
# from .. import LOGGER
#
#
# def generate_scans(t_start: Time, coord0: SkyCoord, location: EarthLocation,
#                    n_scan: int, t_total: float, min_elevation: float = 20.,
#                    min_gap_scan: float = 120.,
#                    partial_scans_allowed: bool = False) -> List['Scan']:
#         """
#         Calculates scan_times for the Observation for a particular pointing
#         centre, from a specific location
#
#         Parameters
#         ----------
#         t_start
#             Start time of first scan
#         coord0
#             Celestial coordinate to observe
#         location
#             Earth location from which to observe
#         n_scan
#             Number of scans
#         t_total
#             Total time of scans [s]
#         min_elevation
#             Minimum elevation of telescope [deg]. Default is 20deg
#         min_gap_scan
#             Minimum time between consecutive scans [s]. Default is 120s
#         partial_scans_allowed
#             Whether to allow scans to be broken across minimum elevation
#             boundaries. Default is False
#         Returns
#         -------
#         List[Scan]
#             List of calculated Scan instances
#         """
#         from ..physics import astronomy as ast
#
#         # Compute scan times
#         scans = ast.scan_times(
#             t_start, coord0, location, n_scan, t_total, min_elevation,
#             min_gap_scan, partial_scans_allowed=partial_scans_allowed
#         )
#
#         return [Scan(start, end) for start, end in scans]
#
#
# @dataclass
# class Field:
#     """Class for holding observational field information"""
#     _ra0: str
#     _dec0: str
#     _frame: str
#     nx: int
#     ny: int
#     cdelt: float  # [deg]
#
#     @property
#     def coord0(self) -> SkyCoord:
#         """Pointing centre as a SkyCoord instance"""
#         import astropy.units as u
#
#         return SkyCoord(self._ra0, self._dec0, frame=self._frame,
#                         unit=(u.hourangle, u.deg))
#
#     @property
#     def fov(self) -> Tuple[float, float]:
#         """Field of view in x and y as a 2-tuple in deg"""
#         return self.nx * self.cdelt, self.ny * self.cdelt
#
#     @property
#     def area(self):
#         """Total area of field of view in deg^2"""
#         return np.prod(self.fov)
#
#
# @dataclass(frozen=True)
# class Scan:
#     """Class for an observational scan"""
#     start: Time
#     end: Time
#
#     @property
#     def duration(self):
#         """Scan duration in seconds"""
#         # Avoid floating point discrepancies with round
#         return round((self.end - self.start).to_value('s'), 1)
#
#     def __str__(self):
#         s = "{0} from {1} to {2} (t_scan = {3:.0f}s)".format(
#             self.__class__.__name__,
#             self.start.strftime('%H:%M:%S-%d%b%Y').upper(),
#             self.end.strftime('%H:%M:%S-%d%b%Y').upper(),
#             self.duration
#         )
#         return s
#
#     def __repr__(self):
#         ast_time_template = f'Time({{}}, scale={{}}, format={{}})'
#         start = ast_time_template.format(repr(self.start.value),
#                                          repr(self.start.scale),
#                                          repr(self.start.format))
#
#         end = ast_time_template.format(repr(self.end.value),
#                                        repr(self.end.scale),
#                                        repr(self.end.format))
#
#         s = "{0}(start_time={1}, end_time={2})".format(
#             self.__class__.__name__, start, end
#         )
#
#         return s
#
#
# class ObservationCfg:
#     """
#     Class containing all information and methods pertaining to a full synthetic
#     observation of variable numbers of scans
#     """
#     t_start: Time  # start time of observation
#     t_total: int  # [s]
#     n_scan: int  # [s]
#     min_gap_scan: int  # [s]
#     min_elevation: int  # [deg]
#
#     def scan_times(
#             self, coord0: SkyCoord, location: EarthLocation,
#             partial_scans_allowed: bool = False
#         """
#         Calculates scan_times for the Observation for a particular pointing
#         centre, from a specific location
#
#         Parameters
#         ----------
#         coord0
#             Celestial coordinate to observe
#         location
#             Earth location from which to observe
#         partial_scans_allowed
#             Whether to allow scans to be broken across minimum elevation
#             boundaries. Default is False
#
#         Returns
#         -------
#         Tuple containing two-tuples of (astropy.time.Time, astropy.time.Time)
#         representing scan start/end times
#         """
#         from ..physics import astronomy as ast
#
#         # Compute scan times
#         scans = ast.scan_times(
#             self.t_start, coord0,
#             location, self.n_scan,
#             self.t_total, self.min_elevation,
#             self.min_gap_scan, partial_scans_allowed=partial_scans_allowed
#         )
#
#         return tuple(scans)
#
#
# class Observation:
#     """
#     Class for an entire observational run incorporating multiple scans
#
#     Methods
#     -------
#     add_scan(new_scans)
#         Add a Scan instance (or list of Scan instances) to Observation's list of
#         Scans
#
#     concat_scan_measurement_sets(out_ms)
#         Concatenate all available scan measurement sets into one (from products)
#
#     create_beam_pattern(scan, beam_fits, resample=1, template=None)
#         Create the station primary beam pattern for a particular scan
#
#     execute_scan(scan, scan_out_ms)
#         Conduct synthetic observing scan using oskar's sim_interferometer task
#
#     generate_scan_seed(inital_seed, scan)
#         Generates fresh seed on basis of scan number and initial seed
#
#     get_scan_tec_screen_slice(tecscreen, scan, scan_tec_fits)
#         Get the TEC screen slice relevant to a scan in the observation
#
#     n_scan(scan)
#         Get Scan's number in terms of chronological order. Zero-indexed
#
#     @staticmethod
#     pb_multiply(in_image, pb, out_fitsfile, cellscal)
#         See farm.miscellaneous.image_functions.pb_multiply
#
#     Attributes
#     ----------
#     cfg: FarmConfiguration
#         farm.config.FarmConfiguration instance
#
#     scans: List[Scan]
#         List of Scan instances attributed to this Observation
#
#     products: dict
#         Dict of products whose keys are Scan instances, and values are dict
#         instances, whose keys are each of Observation._PRODUCT_TYPES and values
#         are full paths to each data product
#
#     total_time: float
#         Total time of observation [s]
#
#     n_scans: int
#         Number of scans in this Observation
#
#     """
#     # from ..calibration.tec import TECScreen
#     # from ..config import FarmConfiguration
#
#     _PRODUCT_TYPES = ('image', 'PB', 'MS', 'TEC', 'seed')
#
#     def __init__(self, config: 'FarmConfiguration',
#                  scans: Union[List[Scan], None] = None):
#         self._config = config
#         self._scans = scans if scans else []
#         self._products = {}
#
#     @property
#     def cfg(self):
#         """FarmConfiguration instance"""
#         return self._config
#
#     @property
#     def products(self):
#         """Data/internal products of the observation"""
#         return self._products
#
#     @products.setter
#     def products(self, new_products):
#         self._products = new_products
#
#     @property
#     def scans(self) -> List[Scan]:
#         """List of scans in observation"""
#         return self._scans
#
#     def add_scan(self, new_scans):
#         """Add a scan"""
#         from ..miscellaneous import error_handling as errh
#
#         if isinstance(new_scans, Iterable):
#             for scan in new_scans:
#                 self.add_scan(scan)
#         else:
#             if not isinstance(new_scans, Scan):
#                 errh.raise_error(TypeError,
#                                  "Can only add Scan instances to "
#                                  "Observation instance, not "
#                                  f"{type(new_scans)} instance")
#             self._scans.append(new_scans)
#             self._scans = sorted(self._scans, key=lambda x: x.start)
#
#             self._products[new_scans] = {}
#             for product_type in self._PRODUCT_TYPES:
#                 self._products[new_scans][product_type] = None
#
#     @property
#     def total_time(self) -> float:
#         """Total time on source, over the course of the observation [s]"""
#         return sum([scan.duration for scan in self.scans])
#
#     @property
#     def n_scans(self) -> int:
#         """Number of scans within the observation"""
#         return len(self.scans)
#
#     def n_scan(self, scan: Scan):
#         """Sequential scan number for a particular scan"""
#         from ..miscellaneous import error_handling as errh
#
#         if scan not in self.scans:
#             errh.raise_error(ValueError,
#                              "Please add {scan} to Observation instance scans "
#                              "using add_scan method")
#
#         return self.scans.index(scan)
#
#     def create_beam_pattern(self, scan: Scan,
#                             beam_fits: pathlib.Path,
#                             resample: int = 1,
#                             template: Optional[pathlib.Path] = None):
#         """
#         Create the station primary beam pattern for a particular scan
#
#         Parameters
#         ----------
#         scan
#             The scan instance to create the beam pattern for. Must have been
#             previously been added to the Observation instances scans
#         beam_fits
#             Full path to save the station beam to (.fits)
#         resample
#             Resampling factor e.g. a value of 2 will have pixels twice as coarse
#             and half the number of pixels in x and y, as specified in the
#             configuration. This can be used to speed up the creation of the
#             station beam pattern through oskar.
#         template
#             Template image to regrid primary beam to, if required
#         Raises
#         -------
#         ValueError
#             If scan has not been added to the Observation instance's scans
#         """
#         from ..miscellaneous import generate_random_chars as grc
#         from ..miscellaneous import error_handling as errh
#         from ..software.oskar import run_oskar_sim_beam_pattern
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         LOGGER.info(f"Creating primary beam for scan #{self.n_scan(scan)}")
#
#         # Assign filename of output beam from oskar's sim_beam_pattern to
#         # variable, oskar_out_beam_fname
#         beam_sfx = '_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I'
#         beam_root = self.cfg.root_name.append(f"_{grc(10)}")
#         beam_name = beam_root.append(beam_sfx)
#         oskar_out_beam_fname = beam_name.append('.fits')
#
#         # Create beam pattern as .fits cube using oskar's sim_beam_pattern task
#         self.cfg.set_oskar_sim_beam_pattern("beam_pattern/root_path", beam_root)
#         self.cfg.set_oskar_sim_beam_pattern("observation/start_time_utc",
#                                             scan.start)
#         self.cfg.set_oskar_sim_beam_pattern("observation/length", scan.duration)
#
#         # Resample if needed
#         nx, ny = self.cfg.field.nx, self.cfg.field.ny
#         cdelt = self.cfg.field.cdelt
#
#         if resample != 1:
#             nx = self.cfg.field.nx // resample + 1
#             ny = self.cfg.field.nx // resample + 1
#             self.cfg.set_oskar_sim_beam_pattern("beam_pattern/beam_image/size",
#                                                 nx)
#             cdelt = self.cfg.field.fov[0] / nx
#
#         scan_sbeam_ini = self.cfg.sbeam_ini.with_name(
#             self.cfg.sbeam_ini.name.replace(
#                 self.cfg.sbeam_ini.suffix,
#                 f"_scan{self.n_scan(scan)}{self.cfg.sbeam_ini.suffix}"
#             )
#         )
#
#         shutil.copyfile(self.cfg.sbeam_ini, scan_sbeam_ini)
#         LOGGER.info(f"Running oskar_sim_beam_pattern from {scan_sbeam_ini}")
#         run_oskar_sim_beam_pattern(scan_sbeam_ini)
#
#         with fits.open(oskar_out_beam_fname) as beam_hdu:
#             bmout_hdu = fits.PrimaryHDU(beam_hdu[0].data[0, ...])
#             bmout_hdu.header.set('CTYPE1', 'RA---SIN')
#             bmout_hdu.header.set('CTYPE2', 'DEC--SIN')
#             bmout_hdu.header.set('CTYPE3', 'FREQ    ')
#             bmout_hdu.header.set('CRVAL1', self.cfg.field.coord0.ra.deg)
#             bmout_hdu.header.set('CRVAL2', self.cfg.field.coord0.dec.deg)
#             bmout_hdu.header.set('CRVAL3', self.cfg.correlator.freq_min)
#             bmout_hdu.header.set('CRPIX1', (nx + 1) / 2)
#             bmout_hdu.header.set('CRPIX2', (ny + 1) / 2)
#             bmout_hdu.header.set('CRPIX3', 1)
#             bmout_hdu.header.set('CDELT1', -cdelt)
#             bmout_hdu.header.set('CDELT2', cdelt)
#             bmout_hdu.header.set('CDELT3', self.cfg.correlator.freq_inc)
#             bmout_hdu.header.set('CUNIT1', 'deg     ')
#             bmout_hdu.header.set('CUNIT2', 'deg     ')
#             bmout_hdu.header.set('CUNIT3', 'Hz      ')
#             bmout_hdu.writeto(beam_fits, overwrite=True)
#
#         if not beam_fits.exists():
#             errh.raise_error(FileNotFoundError, f"{beam_fits} not written")
#
#         LOGGER.info(f"Successfully written primary beam for scan "
#                     f"#{self.n_scan(scan)} to {beam_fits}")
#
#         self._products[scan]['PB'] = beam_fits
#         oskar_out_beam_fname.unlink()
#
#         # Reset oskar settings to defaults in case of resampling
#         if resample != 1:
#             self.cfg.set_oskar_sim_beam_pattern("beam_pattern/beam_image/size",
#                                                 self.cfg.field.nx)
#
#         if template:
#             from ..miscellaneous.image_functions import regrid_fits
#             regrid_fits(beam_fits, template, inplace=True)
#
#     def generate_scan_seed(self, initial_seed: int, scan: int) -> int:
#         """Generates fresh seed on basis of scan number and initial seed"""
#         idx_scan = self.n_scan(scan)
#         seed = int(initial_seed + idx_scan * (3 - 5 ** 0.5) * 180.)
#         self.products[scan]['seed'] = seed
#         LOGGER.info(f"For scan #{self.n_scan(scan)}, {seed} generated for "
#                     f"random number generator seed")
#         return seed
#
#     def get_scan_tec_screen_slice(self, tecscreen: 'TECScreen', scan: Scan,
#                                   scan_tec_fits: pathlib.Path):
#         """
#         Get the TEC screen slice relevant to a scan in the observation
#
#         Parameters
#         ----------
#         tecscreen
#             TECScreen instance from which to extract slice
#         scan
#             Scan over which to extract TEC screen
#         scan_tec_fits
#             Full path to resulting TEC screen .fits image\
#
#         Raises
#         -------
#         ValueError
#             If scan is not listed in Observation instance's scans
#         """
#         from ..miscellaneous import error_handling as errh
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         tecscreen.extract_tec_screen_slice(
#             scan_tec_fits, scan.start, scan.end
#         )
#
#         with fits.open(scan_tec_fits, memmap=True) as hdul:
#             hdul[0].data *= self.cfg.calibration.tec.err
#             hdul.writeto(scan_tec_fits, overwrite=True)
#
#         self.products[scan]['TEC'] = scan_tec_fits
#
#         return scan_tec_fits
#
#     def execute_scan(self, scan: Scan, scan_out_ms: pathlib.Path):
#         """
#         Conduct synthetic observing scan using oskar's sim_interferometer task
#
#         Parameters
#         ----------
#         scan
#             Scan to conduct synthetic observation on
#         scan_out_ms
#             Full path to output measurement set
#
#         Raises
#         -------
#         FileNotFoundError
#             If measurement set was not produced by oskar
#         """
#         from ..miscellaneous import error_handling as errh
#         from ..software import oskar
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         # Adjust oskar's sim_interferometer settings in .ini files
#         self.cfg.set_oskar_sim_interferometer(
#             'sky/oskar_sky_model/file', self.cfg.oskar_sky_model_file
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/start_time_utc',
#             scan.start.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-2]
#         )
#
#         if self.cfg.calibration.tec:
#             self.cfg.set_oskar_sim_interferometer(
#                 'telescope/external_tec_screen/input_fits_file',
#                 self.products[scan]['TEC']
#             )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'interferometer/ms_filename', scan_out_ms
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/length', format(scan.duration, '.1f')
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/num_time_steps',
#             int(scan.duration // self.cfg.correlator.t_int)
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'interferometer/noise/seed', self.products[scan]['seed']
#         )
#
#         scan_ini = self.cfg.sinterferometer_ini.with_name(
#                         self.cfg.sinterferometer_ini.name.replace(
#                             self.cfg.sinterferometer_ini.suffix,
#                             f"_scan{self.n_scan(scan)}"
#                             f"{self.cfg.sinterferometer_ini.suffix}"
#                         )
#                     )
#         shutil.copyfile(self.cfg.sinterferometer_ini, scan_ini)
#         LOGGER.info(f"Running oskar_sim_interferometer from {scan_ini}")
#         oskar.run_oskar_sim_interferometer(scan_ini)
#
#         if not scan_out_ms.exists():
#             errh.raise_error(FileNotFoundError,
#                              f"Output measurement set from oskar, {scan_out_ms}"
#                              f", not produced/found")
#
#         self._products[scan]['ms'] = scan_out_ms
#
#     def concat_scan_measurement_sets(self, out_ms: pathlib.Path):
#         """
#         Concatenate all available scan measurement sets into one
#
#         Parameters
#         ----------
#         out_ms
#             Full path of final produced measurement set
#         """
#         from ..software import casa
#
#         measurement_sets = [str(self.products[s]['MS']) for s in self.scans]
#
#         casa.tasks.concat(vis=measurement_sets, concatvis=out_ms, timesort=True)
#
#     @staticmethod
#     def pb_multiply(in_image: pathlib.Path, pb: pathlib.Path,
#                     out_fitsfile: pathlib.Path, cellscal: str = 'CONSTANT'):
#         from ..miscellaneous.image_functions import pb_multiply
#
#         pb_multiply(in_image=in_image, pb=pb,
#                     out_fitsfile=out_fitsfile, cellscal=cellscal)
#
#     def __str__(self):
#         s = (f"Observation of "
#              f"{self.cfg.field.coord0.ra.to_string('hour')}, "
#              f"{self.cfg.field.coord0.dec.to_string('deg')}, totalling "
#              f"{self.total_time / 60 ** 2.:.2f} hours split over "
#              f"{self.n_scans} scan(s):\n - ")
#         s += '\n -'.join([str(scan) for scan in self.scans])
#
#         return s
#
#
# class ObservationNew:
#     """
#     Class for an entire observational run incorporating multiple scans
#
#     Methods
#     -------
#     add_scan(new_scans)
#         Add a Scan instance (or list of Scan instances) to Observation's list of
#         Scans
#
#     concat_scan_measurement_sets(out_ms)
#         Concatenate all available scan measurement sets into one (from products)
#
#     create_beam_pattern(scan, beam_fits, resample=1, template=None)
#         Create the station primary beam pattern for a particular scan
#
#     execute_scan(scan, scan_out_ms)
#         Conduct synthetic observing scan using oskar's sim_interferometer task
#
#     generate_scan_seed(inital_seed, scan)
#         Generates fresh seed on basis of scan number and initial seed
#
#     get_scan_tec_screen_slice(tecscreen, scan, scan_tec_fits)
#         Get the TEC screen slice relevant to a scan in the observation
#
#     n_scan(scan)
#         Get Scan's number in terms of chronological order. Zero-indexed
#
#     @staticmethod
#     pb_multiply(in_image, pb, out_fitsfile, cellscal)
#         See farm.miscellaneous.image_functions.pb_multiply
#
#     Attributes
#     ----------
#     field: Field
#         farm.observing.Field instance representing observational field
#
#     scans: List[Scan]
#         List of Scan instances attributed to this Observation
#
#     products: dict
#         Dict of products whose keys are Scan instances, and values are dict
#         instances, whose keys are each of Observation._PRODUCT_TYPES and values
#         are full paths to each data product
#
#     total_time: float
#         Total time of observation [s]
#
#     n_scans: int
#         Number of scans in this Observation
#
#     """
#     # from ..calibration.tec import TECScreen
#     # from ..config import FarmConfiguration
#
#     _PRODUCT_TYPES = ('image', 'PB', 'MS', 'TEC', 'seed')
#
#     def __init__(self, field: Field, telescope: 'Telescope',
#                  correlator: 'Correlator',
#                  scans: Union[List[Scan], None] = None):
#         """
#         Parameters
#         ----------
#         field
#             farm.observing.Field instance for the observational field
#         telescope
#             farm.observing.Telescope instance for the observation
#         correlator
#             farm.observing.correlator instance for the observational setup
#         """
#         self.field: Field = field
#         self.telescope = telescope
#         self.correlator = correlator
#
#         self._scans = []
#         self._products = {}
#
#     @property
#     def t_start(self) -> Time:
#         """Observation start time"""
#         return min(scan.start for scan in self.scans) if self.scans else None
#
#     @property
#     def t_end(self) -> Time:
#         """Observation start time"""
#         return max(scan.end for scan in self.scans) if self.scans else None
#
#     @property
#     def total_time(self) -> float:
#         """Total time on source, over the course of the observation [s]"""
#         return sum([scan.duration for scan in self.scans]) if self.scans else 0.
#
#     @property
#     def products(self) -> dict:
#         """Data/internal products of the observation"""
#         return self._products
#
#     @products.setter
#     def products(self, new_products):
#         self._products = new_products
#
#     @property
#     def scans(self) -> List[Scan]:
#         """List of scans in observation"""
#         return self._scans
#
#     def add_scan(self, new_scans: Union[List[Scan], Scan]):
#         """Add a scan"""
#         from ..miscellaneous import error_handling as errh
#
#         if isinstance(new_scans, Iterable):
#             for scan in new_scans:
#                 self.add_scan(scan)
#         else:
#             if not isinstance(new_scans, Scan):
#                 errh.raise_error(TypeError,
#                                  "Can only add Scan instances to "
#                                  "Observation instance, not "
#                                  f"{type(new_scans)} instance")
#             self._scans.append(new_scans)
#             self._scans = sorted(self._scans, key=lambda x: x.start)
#
#             self._products[new_scans] = {}
#             for product_type in self._PRODUCT_TYPES:
#                 self._products[new_scans][product_type] = None
#
#     @property
#     def n_scans(self) -> int:
#         """Number of scans within the observation"""
#         return len(self.scans)
#
#     def n_scan(self, scan: Scan) -> int:
#         """Sequential scan number for a particular scan"""
#         from ..miscellaneous import error_handling as errh
#
#         if scan not in self.scans:
#             errh.raise_error(ValueError,
#                              "Please add {scan} to Observation instance scans "
#                              "using add_scan method")
#
#         return self.scans.index(scan)
#
#     def create_beam_pattern(self, scan: Scan,
#                             beam_fits: pathlib.Path,
#                             resample: int = 1,
#                             template: Optional[pathlib.Path] = None):
#         """
#         Create the station primary beam pattern for a particular scan
#
#         Parameters
#         ----------
#         scan
#             The scan instance to create the beam pattern for. Must have been
#             previously been added to the Observation instances scans
#         beam_fits
#             Full path to save the station beam to (.fits)
#         resample
#             Resampling factor e.g. a value of 2 will have pixels twice as coarse
#             and half the number of pixels in x and y, as specified in the
#             configuration. This can be used to speed up the creation of the
#             station beam pattern through oskar.
#         template
#             Template image to regrid primary beam to, if required
#         Raises
#         -------
#         ValueError
#             If scan has not been added to the Observation instance's scans
#         """
#         from ..miscellaneous import generate_random_chars as grc
#         from ..miscellaneous import error_handling as errh
#         from ..software.oskar import run_oskar_sim_beam_pattern
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         LOGGER.info(f"Creating primary beam for scan #{self.n_scan(scan)}")
#
#         # Assign filename of output beam from oskar's sim_beam_pattern to
#         # variable, oskar_out_beam_fname
#         beam_sfx = '_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I'
#         beam_root = self.cfg.root_name.append(f"_{grc(10)}")
#         beam_name = beam_root.append(beam_sfx)
#         oskar_out_beam_fname = beam_name.append('.fits')
#
#         # Create beam pattern as .fits cube using oskar's sim_beam_pattern task
#         self.cfg.set_oskar_sim_beam_pattern("beam_pattern/root_path", beam_root)
#         self.cfg.set_oskar_sim_beam_pattern("observation/start_time_utc",
#                                             scan.start)
#         self.cfg.set_oskar_sim_beam_pattern("observation/length", scan.duration)
#
#         # Resample if needed
#         nx, ny = self.field.nx, self.field.ny
#         cdelt = self.field.cdelt
#
#         if resample != 1:
#             nx = self.field.nx // resample + 1
#             ny = self.field.nx // resample + 1
#             self.cfg.set_oskar_sim_beam_pattern("beam_pattern/beam_image/size",
#                                                 nx)
#             cdelt = self.field.fov[0] / nx
#
#         scan_sbeam_ini = self.cfg.sbeam_ini.with_name(
#             self.cfg.sbeam_ini.name.replace(
#                 self.cfg.sbeam_ini.suffix,
#                 f"_scan{self.n_scan(scan)}{self.cfg.sbeam_ini.suffix}"
#             )
#         )
#
#         shutil.copyfile(self.cfg.sbeam_ini, scan_sbeam_ini)
#         LOGGER.info(f"Running oskar_sim_beam_pattern from {scan_sbeam_ini}")
#         run_oskar_sim_beam_pattern(scan_sbeam_ini)
#
#         with fits.open(oskar_out_beam_fname) as beam_hdu:
#             bmout_hdu = fits.PrimaryHDU(beam_hdu[0].data[0, ...])
#             bmout_hdu.header.set('CTYPE1', 'RA---SIN')
#             bmout_hdu.header.set('CTYPE2', 'DEC--SIN')
#             bmout_hdu.header.set('CTYPE3', 'FREQ    ')
#             bmout_hdu.header.set('CRVAL1', self.field.coord0.ra.deg)
#             bmout_hdu.header.set('CRVAL2', self.field.coord0.dec.deg)
#             bmout_hdu.header.set('CRVAL3', self.cfg.correlator.freq_min)
#             bmout_hdu.header.set('CRPIX1', (nx + 1) / 2)
#             bmout_hdu.header.set('CRPIX2', (ny + 1) / 2)
#             bmout_hdu.header.set('CRPIX3', 1)
#             bmout_hdu.header.set('CDELT1', -cdelt)
#             bmout_hdu.header.set('CDELT2', cdelt)
#             bmout_hdu.header.set('CDELT3', self.cfg.correlator.freq_inc)
#             bmout_hdu.header.set('CUNIT1', 'deg     ')
#             bmout_hdu.header.set('CUNIT2', 'deg     ')
#             bmout_hdu.header.set('CUNIT3', 'Hz      ')
#             bmout_hdu.writeto(beam_fits, overwrite=True)
#
#         if not beam_fits.exists():
#             errh.raise_error(FileNotFoundError, f"{beam_fits} not written")
#
#         LOGGER.info(f"Successfully written primary beam for scan "
#                     f"#{self.n_scan(scan)} to {beam_fits}")
#
#         self._products[scan]['PB'] = beam_fits
#         oskar_out_beam_fname.unlink()
#
#         # Reset oskar settings to defaults in case of resampling
#         if resample != 1:
#             self.cfg.set_oskar_sim_beam_pattern("beam_pattern/beam_image/size",
#                                                 self.field.nx)
#
#         if template:
#             from ..miscellaneous.image_functions import regrid_fits
#             regrid_fits(beam_fits, template, inplace=True)
#
#     def generate_scan_seed(self, initial_seed: int, scan: int) -> int:
#         """Generates fresh seed on basis of scan number and initial seed"""
#         idx_scan = self.n_scan(scan)
#         seed = int(initial_seed + idx_scan * (3 - 5 ** 0.5) * 180.)
#         self.products[scan]['seed'] = seed
#         LOGGER.info(f"For scan #{self.n_scan(scan)}, {seed} generated for "
#                     f"random number generator seed")
#         return seed
#
#     def get_scan_tec_screen_slice(self, tecscreen: 'TECScreen', scan: Scan,
#                                   scan_tec_fits: pathlib.Path):
#         """
#         Get the TEC screen slice relevant to a scan in the observation
#
#         Parameters
#         ----------
#         tecscreen
#             TECScreen instance from which to extract slice
#         scan
#             Scan over which to extract TEC screen
#         scan_tec_fits
#             Full path to resulting TEC screen .fits image\
#
#         Raises
#         -------
#         ValueError
#             If scan is not listed in Observation instance's scans
#         """
#         from ..miscellaneous import error_handling as errh
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         tecscreen.extract_tec_screen_slice(
#             scan_tec_fits, scan.start, scan.end
#         )
#
#         with fits.open(scan_tec_fits, memmap=True) as hdul:
#             hdul[0].data *= self.cfg.calibration.tec.err
#             hdul.writeto(scan_tec_fits, overwrite=True)
#
#         self.products[scan]['TEC'] = scan_tec_fits
#
#         return scan_tec_fits
#
#     def execute_scan(self, scan: Scan, scan_out_ms: pathlib.Path):
#         """
#         Conduct synthetic observing scan using oskar's sim_interferometer task
#
#         Parameters
#         ----------
#         scan
#             Scan to conduct synthetic observation on
#         scan_out_ms
#             Full path to output measurement set
#
#         Raises
#         -------
#         FileNotFoundError
#             If measurement set was not produced by oskar
#         """
#         from ..miscellaneous import error_handling as errh
#         from ..software import oskar
#
#         if scan not in self.scans:
#             msg = f"{scan} not added to Observation's scans. Use add_scan"
#             errh.raise_error(ValueError, msg)
#
#         # Adjust oskar's sim_interferometer settings in .ini files
#         self.cfg.set_oskar_sim_interferometer(
#             'sky/oskar_sky_model/file', self.cfg.oskar_sky_model_file
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/start_time_utc',
#             scan.start.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-2]
#         )
#
#         if self.cfg.calibration.tec:
#             self.cfg.set_oskar_sim_interferometer(
#                 'telescope/external_tec_screen/input_fits_file',
#                 self.products[scan]['TEC']
#             )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'interferometer/ms_filename', scan_out_ms
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/length', format(scan.duration, '.1f')
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'observation/num_time_steps',
#             int(scan.duration // self.cfg.correlator.t_int)
#         )
#
#         self.cfg.set_oskar_sim_interferometer(
#             'interferometer/noise/seed', self.products[scan]['seed']
#         )
#
#         scan_ini = self.cfg.sinterferometer_ini.with_name(
#                         self.cfg.sinterferometer_ini.name.replace(
#                             self.cfg.sinterferometer_ini.suffix,
#                             f"_scan{self.n_scan(scan)}"
#                             f"{self.cfg.sinterferometer_ini.suffix}"
#                         )
#                     )
#         shutil.copyfile(self.cfg.sinterferometer_ini, scan_ini)
#         LOGGER.info(f"Running oskar_sim_interferometer from {scan_ini}")
#         oskar.run_oskar_sim_interferometer(scan_ini)
#
#         if not scan_out_ms.exists():
#             errh.raise_error(FileNotFoundError,
#                              f"Output measurement set from oskar, {scan_out_ms}"
#                              f", not produced/found")
#
#         self._products[scan]['ms'] = scan_out_ms
#
#     def concat_scan_measurement_sets(self, out_ms: pathlib.Path):
#         """
#         Concatenate all available scan measurement sets into one
#
#         Parameters
#         ----------
#         out_ms
#             Full path of final produced measurement set
#         """
#         from ..software import casa
#
#         measurement_sets = [str(self.products[s]['MS']) for s in self.scans]
#
#         casa.tasks.concat(vis=measurement_sets, concatvis=out_ms, timesort=True)
#
#     @staticmethod
#     def pb_multiply(in_image: pathlib.Path, pb: pathlib.Path,
#                     out_fitsfile: pathlib.Path, cellscal: str = 'CONSTANT'):
#         from ..miscellaneous.image_functions import pb_multiply
#
#         pb_multiply(in_image=in_image, pb=pb,
#                     out_fitsfile=out_fitsfile, cellscal=cellscal)
#
#     def __str__(self):
#         s = (f"Observation of "
#              f"{self.field.coord0.ra.to_string('hour')}, "
#              f"{self.field.coord0.dec.to_string('deg')}, totalling "
#              f"{self.total_time / 60 ** 2.:.2f} hours split over "
#              f"{self.n_scans} scan(s):\n - ")
#         s += '\n -'.join([str(scan) for scan in self.scans])
#
#         return s
#

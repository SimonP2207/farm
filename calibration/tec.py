"""
Module handling direction-dependent errors in the form of total electron counts
(TEC) through the atmosphere via the TECScreen class.

Classes
-------
_ArScreens (private)
    Implementation of ARatmospy.ArScreens.ArScreens class
TECScreen
    Handles logic related to TEC screen generation and book-keeping
"""
import warnings
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import numpy.typing as npt
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import ARatmospy

from ..miscellaneous import error_handling as errh
from ..miscellaneous.decorators import convert_str_to_path
from .. import LOGGER


def extent_m(subtended_angle: float, dist: float) -> float:
    """
    Return the physical extent [m] of a length subtending an angle,
    subtended_angle [deg], located a distance from the observer, dist [m]
    """
    return 2. * dist * np.tan(np.radians(subtended_angle / 2.))

def extent_deg(physical_extent: float, dist: float) -> float:
    """
    Return the angular extent [deg] of the subtended angle of a length,
    physical_extent [m], located a distance from the observer, dist [m]
    """
    return np.degrees(2. * np.arctan(physical_extent / (2. * dist)))


class _ArScreens(ARatmospy.ArScreens.ArScreens):
    """
    FARM adaptation of ARatmospy.ArScreens.ArScreens class (used to generate
    atmospheric phase screens using an autoregressive process to add stochastic
    noise to an otherwise frozen flow). Purpose of this adaptation is to retain
    memory of the parameters used to instantiate an _ArScreens instance for
    future use. After instantiation of an _ArScreens instance, parameters can't
    be changed.
    """

    @classmethod
    def create_from_params(cls, t_int: float, pixel_m: float, fov: float,
                           bmax: float, layer_params: List[List],
                           alpha_mag: float, rseed: int) -> '_ArScreens':
        """
        Create the _ArScreens instance from typically well-known parameters

        Parameters
        ----------
        t_int
            Time between frames of TEC screen
        pixel_m
            Desired pixel size [m]
        fov
            Minimum field of view for screen
        bmax
            Maximum baseline length [m]
        layer_params
            Array describing each layer of the atmosphere to be modeled. Each
            row contains a tuple of (r0 [m], velocity [m/s], direction [deg],
            altitude [m])
        alpha_mag
            Magnitude of autoregressive parameter. (1-alpha_mag) is the fraction
            of the phase from the prior time step that is "forgotten" and
            replaced by Gaussian noise
        rseed
            Random number generator seed to use

        Returns
        -------
        TECScreen
            Created TECScreen instance
        """
        # Screen width calculated from altitude of highest screen and field of
        # view, plus padding for the maximum baseline length on each side
        layer_max_height = np.max(np.array(layer_params)[:, -1])
        screen_width_m = extent_m(fov, layer_max_height) + bmax * 2.

        # Round up the screen width to a sensible multiple of the pixel width
        screen_width_m = ((screen_width_m / pixel_m // 100 + 1) * 100 * pixel_m)

        # n and m required to be integers by ARatmospy.create_multilayer_arbase
        n = int(screen_width_m / bmax) + 1 # Sub-apertures across screen
        m = int(bmax / pixel_m) + 1  # Pixels per sub-aperture

        pscale = pixel_m
        rate = t_int ** -1.  # Update rate [Hz]

        return cls(n, m, pscale, rate, layer_params, alpha_mag, rseed)

    def __init__(self, n: int, m: int, pscale: float, rate: float,
                 paramcube: npt.NDArray[float], alpha_mag: float, ranseed: int):
        """
        Parameters
        ----------
        n
            Number of subapertures across the screen
        m
            Number of pixels per subaperature
        pscale
            Pixel scale [m]
        rate
            A0 system rate [Hz]
        paramcube
            Parameter array describing each layer of the atmosphere to be
            modeled. Each row contains a tuple of (r0 [m], velocity [m/s],
            direction [deg], altitude [m])
        alpha_mag
            Magnitude of autoregressive parameter. (1-alpha_mag) is the fraction
            of the phase from the prior time step that is "forgotten" and
            replaced by Gaussian noise
        ranseed
            Random number generator seed
        """
        # To catch 'divide by zero' warnings raised by
        # ARatmospy.create_multilayer_arbase method
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    module='ARatmospy')
            super().__init__(n, m, pscale, rate, paramcube, alpha_mag, ranseed)

        self._n = n
        self._m = m
        self._pscale = pscale
        self._rate = rate
        self._layer_params = paramcube
        self._alpha_mag = alpha_mag
        self._rseed = ranseed

    @property
    def n(self):
        """Number of subapertures across the screen"""
        return self._n

    @property
    def m(self):
        """Number of pixels per subaperature"""
        return self._m

    @property
    def pscale(self):
        """Pixel scale [m]"""
        return self._pscale

    @property
    def rate(self):
        """
        A0 system rate, whereby 1 / rate is the gap (in seconds) between
        consecutive frames of the phase screen [Hz]
        """
        return self._rate

    @property
    def t_int(self):
        """Gap between consecutive frames of the phase screen [s]"""
        return 1. / self.rate

    @property
    def layer_params(self):
        """
        Parameter array describing each layer of the atmosphere to be
        modeled.  Each row contains a tuple of (r0 [m], velocity [m/s],
        direction [deg], altitude [m]) describing the corresponding layer
        """
        return self._layer_params

    @property
    def alpha_mag(self):
        """
        Magnitude of autoregressive parameter, whereby (1-alpha_mag) is the
        fraction of the phase from the prior time step that is "forgotten" and
        replaced by Gaussian noise
        """
        return self._alpha_mag

    @property
    def rseed(self):
        """Random number generator seed"""
        return self._rseed


class TECScreen:
    """
    Class to hold all attributes/methods concerning a complete TEC screen
    generation
    """
    _LAYERS_DATA_HDR = ('r0', 'vel', 'direction', 'altitude')
    _LAYERS_DATA_HDR_UNITS = {
        'r0': 'm', 'vel': 'm/s', 'direction': 'deg', 'altitude': 'm'
    }

    @staticmethod
    @convert_str_to_path('data_file')
    def _layers_from_file(data_file: Union[str, Path]
                          ) -> List[Dict[str, float]]:
        """
        Parse TEC screen layer parameters from a text file. The text file should
        be organised in four columns of as many rows as there are desired
        layers. From left to right, the columns should be 'r0', 'vel',
        'direction', 'altitude'. Lines preceded by a '#' are ignored
        """
        hdr = TECScreen._LAYERS_DATA_HDR
        data = np.loadtxt(data_file, comments='#')

        return [{k: row[i] for i, k in enumerate(hdr)} for row in data]

    @classmethod
    @convert_str_to_path('fitsfile')
    def create_from_fits(cls, fitsfile: Path,
                         t_start: Time) -> 'TECScreen':
        """
        Create a TEC screen instance from a previously created TEC screen .fits

        Parameters
        ----------
        fitsfile
            Full path to TEC screen .fits
        t_start
            Start time

        Returns
        -------
        TECScreen instance. Be aware there will be no _ArScreens attribute on
        the created TECScreen instance
        """
        with fits.open(fitsfile, memmap=True) as hdul:
            hdr = hdul[0].header
            duration = hdr['CDELT3'] * u.s * hdr['NAXIS3']
            t_end = t_start + duration

        tecscreen = cls(
            t_start, t_end, hdr['CDELT3'], hdr['CDELT1'],
            None, None, None, None, None
        )
        tecscreen._cube = fitsfile

        return tecscreen

    def __init__(self, t_start: Time, t_end: Time, t_int: float,
                 pixel_m: float, fov: float, bmax: float,
                 layer_params: Union[str, Path, List[Dict[str, float]]],
                 alpha_mag: float,
                 rseed: int):
        """
        Parameters
        ----------
        t_start
            Start time of the TEC screen
        t_end
            Start time of the TEC screen
        t_int
            Time between frames of TEC screen
        pixel_m
            Desired pixel size [m]
        fov
            Minimum field of view for screen [deg]
        bmax
            Maximum baseline length [m]
        layer_params
            Array describing each layer of the atmosphere to be modeled. Each
            row contains a tuple of (r0 [m], velocity [m/s], direction [deg],
            altitude [m])
        alpha_mag
            Magnitude of autoregressive parameter. (1-alpha_mag) is the fraction
            of the phase from the prior time step that is "forgotten" and
            replaced by Gaussian noise
        rseed
            Random number generator seed to use
        """
        #
        if isinstance(layer_params, (str, Path)):
            layer_params = self._layers_from_file(layer_params)
        try:
            layer_params = np.array(
                [[d[h] for h in self._LAYERS_DATA_HDR] for d in layer_params]
            )
            self._arscreen = _ArScreens.create_from_params(
                t_int, pixel_m, fov, bmax, layer_params, alpha_mag, rseed
            )
        # In case we use the create_from_fits method
        except OSError:
            self._arscreen = None

        self._t_start = t_start
        self._t_end = t_end
        self._times: Tuple = None
        self._cube = None

    @property
    def arscreen(self) -> _ArScreens:
        """farm.calibration.tec._ArScreens instance"""
        return self._arscreen

    @property
    def times(self) -> Tuple[Time]:
        """Tuple of times corresponding to each frame"""
        if self._times is None:
            self._times = []
            t = self.t_start
            while t + self.interval * u.s < self.t_end:
                self._times.append(t)
                t += self.interval * u.s
            self._times = tuple(self._times)

        return self._times

    @property
    def t_start(self) -> Time:
        """Start time to establish phase screen"""
        return self._t_start

    @t_start.setter
    def t_start(self, new_t_start: Time):
        if self._cube is not None:
            errh.raise_error(AttributeError,
                             "After running TECScreen, can't change t_start")
        self._times = None
        self._t_start = new_t_start

    @property
    def t_end(self) -> Time:
        """Final time to stop evolving phase screen"""
        return self._t_end

    @t_end.setter
    def t_end(self, new_t_end: Time):
        if self._cube is not None:
            errh.raise_error(AttributeError,
                             "After running TECScreen, can't change t_end")
        self._times = None
        self._t_end = new_t_end

    @property
    def duration(self) -> float:
        """Total duration over which to evolve phase screen [s]"""
        return (self.t_end - self.t_start).to('second').value

    @property
    def rate(self) -> float:
        """Rate of time sampling [Hz]"""
        if self.arscreen is not None:
            return self.arscreen.rate

        with fits.open(self.cube, memmap=True) as hdul:
            return 1. / hdul[0].header['CDELT3']

    @property
    def interval(self) -> float:
        """Time-interval between frames [s]"""
        return 1. / self.rate

    @property
    def cube(self):
        """Full path to .fits cube produced by _ArScreens instance"""
        return self._cube

    @property
    def rseed(self):
        """Random number generator seed used to instantiate _ArScreens"""
        return self.arscreen.rseed

    def create_tec_screen(self, fitsfile: Path):
        """
        Create differential Total Electron count (dTEC) screen as a 4-D .fits
        image (x, y, time, frequency)

        Parameters
        ----------
        fitsfile
            Full path to write TEC screen to
        """
        from ..miscellaneous.decorators import time_it

        if self._cube is not None:
            LOGGER.info(f"TEC screen previously created -> {self._cube}")
            return None

        num_times = int(self.duration / self.arscreen.t_int) + 1
        num_pix = self.arscreen.n * self.arscreen.m

        LOGGER.info(f"Started generating phase screen over {num_times} time "
                    f"intervals with grid size {num_pix} \u00D7 {num_pix} "
                    f"pixels")
        comp_time, _ = time_it(self.arscreen.run)(num_times)
        LOGGER.info(f"Finished generating phase screen in {comp_time:.0f}s")

        # Convert to TEC, see:
        # https://ska-telescope.gitlab.io/sim/oskar/python/example_ionosphere.html?highlight=tec
        freq = 1e8
        phase2tec = -freq / 8.44797245e9

        # hdr = fits.Header()
        data = np.zeros((100, 100, 100), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data)
        hdr = hdu.header
        hdr.set('SIMPLE', True)
        hdr.set('BITPIX', -32)
        hdr.set('NAXIS', 4)
        hdr.set('NAXIS1', num_pix)
        hdr.set('NAXIS2', num_pix)
        hdr.set('NAXIS3', num_times)
        hdr.set('NAXIS4', 1)
        hdr.set('CTYPE1', 'XX')
        hdr.set('CTYPE2', 'YY')
        hdr.set('CTYPE3', 'TIME')
        hdr.set('CTYPE4', 'FREQ')
        hdr.set('CRVAL1', 0.0)
        hdr.set('CRVAL2', 0.0)
        hdr.set('CRVAL3', 0.0)
        hdr.set('CRVAL4', freq)
        hdr.set('CRPIX1', num_pix // 2 + 1)
        hdr.set('CRPIX2', num_pix // 2 + 1)
        hdr.set('CRPIX3', 1.0)
        hdr.set('CRPIX4', 1.0)
        hdr.set('CDELT1', self.arscreen.pscale)
        hdr.set('CDELT2', self.arscreen.pscale)
        hdr.set('CDELT3', self.arscreen.t_int)
        hdr.set('CDELT4', 1.0)
        hdr.set('CUNIT1', 'm')
        hdr.set('CUNIT2', 'm')
        hdr.set('CUNIT3', 's')
        hdr.set('CUNIT4', 'Hz')
        hdr.set('LATPOLE', 90.0)
        hdr.set('MJDREF', 0.0)
        hdr.set('BTYPE', 'dTEC')

        while len(hdr) % 36 != 0:
            # Add blank cards to the end of header until header block length
            # reaches a multiple of 36 (fits headers required to be written in
            # blocks of 2880 bytes
            hdr.append(useblanks=True)

        hdr.tofile(fitsfile, overwrite=True)
        shape = tuple(hdr[f'NAXIS{ii}'] for ii in range(1, hdr['NAXIS'] + 1))
        with open(fitsfile, 'rb+') as fobj:
            fobj.seek(len(hdr.tostring()) +
                      (np.product(shape) * np.abs(hdr['BITPIX'] // 8)) - 1)
            fobj.write(b'\0')

        # FIXME: Catches warnings regarding .fits file truncation raised by
        #  astropy.io.fits.file._File.seek method
        from astropy.utils.exceptions import AstropyUserWarning
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module='astropy',
                                    category=AstropyUserWarning)

            with fits.open(fitsfile, memmap=True, mode='update') as hdul:
                for layer in range(len(self.arscreen.screens)):
                    for i, screen in enumerate(self.arscreen.screens[layer]):
                        hdul[0].data[:, i, ...] += phase2tec * \
                                                   screen[np.newaxis, ...]

        if not fitsfile.exists():
            errh.raise_error(FileNotFoundError,
                             f'TEC Screen not generated ({fitsfile} not '
                             f'written)')

        self._cube = fitsfile.resolve()

    def extract_tec_screen_slice(self, slice_fitsfile: Path,
                                 t_start: Time, t_end: Time):
        """
        Extract relevant time slice from the TEC screen and save as a new .fits
        file

        Parameters
        ----------
        slice_fitsfile
            Full path to save slice to as a .fits file
        t_start
            Start time of slice
        t_end
            End time of slice
        """
        if self.cube is None:
            errh.raise_error(FileNotFoundError,
                             "TEC screen not created, must execute "
                             "create_tec_screen first")

        intervals_in_range = list(filter(lambda x: t_start <= x[1] < t_end,
                                  enumerate(self.times)))
        t_idxs = [_[0] for _ in intervals_in_range]
        LOGGER.info(f"Extracting TEC slice in time range {t_start} - {t_end} "
                    f"of {len(t_idxs)} intervals, from {self._cube}")

        with fits.open(self._cube, memmap=True) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data[:, t_idxs, :, :]

        hdr['NAXIS3'] = len(t_idxs)
        hdr.tofile(slice_fitsfile, overwrite=True)
        shape = tuple(hdr[f'NAXIS{ii}'] for ii in range(1, hdr['NAXIS'] + 1))

        with open(slice_fitsfile, 'rb+') as fobj:
            fobj.seek(len(hdr.tostring()) +
                      (np.product(shape) * np.abs(hdr['BITPIX'] // 8)) -
                      1)
            fobj.write(b'\0')

        with fits.open(slice_fitsfile, memmap=True, mode='update') as hdul:
            hdul[0].data = data

        LOGGER.info(f"TEC slice successfully written to {slice_fitsfile}")

    @property
    def physical_extent(self):
        """Physical extent of the TECScreen [m]"""
        return self.arscreen.m * self.arscreen.n * self.arscreen.pscale

    @property
    def angular_extent(self):
        """Angular extent of the TECScreen [deg]"""
        max_height = np.max(np.array(self.arscreen.layer_params)[:, -1])

        return extent_deg(self.physical_extent, max_height)

    @property
    def summary(self) -> str:
        """Summary of the TECScreen"""
        s_layers = []
        for i, layer in enumerate(self.arscreen.layer_params):
            s_layer = []
            for i2, h in enumerate(self._LAYERS_DATA_HDR):
                unit = self._LAYERS_DATA_HDR_UNITS[h]
                s = '' if i2 else f'Layer {i:02} -> '
                fmt = '.1f' if layer[i2] < 1e3 else '.1e'
                s += f"{h}: {format(layer[i2], fmt)}{unit}"
                s_layer.append(s)
            s_layers.append('; '.join(s_layer))
        s_layers = '\n - '.join(s_layers)

        npix = self.arscreen.m * self.arscreen.n
        s = f"{self.__class__.__name__} of {npix} \u00D7 {npix} pixels " \
            f"({self.arscreen.pscale:.0f}m pixel size) covering " \
            f"{self.physical_extent / 1e3} km\u00B2 " \
            f"({self.angular_extent:.1f}deg\u00B2 at centre) and comprised of" \
            f" {len(self.arscreen.layer_params)} layers whose properties are:" \
            f" \n\n - {s_layers}"

        if self._cube is None:
            s += '\n\ndTECs not yet written to file.'
        else:
            s += f'\n\ndTECs written to {self.cube.resolve()}.'
        return s

import pathlib
import logging
from typing import Union, Optional, List, Tuple

import numpy as np
import numpy.typing as npt
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import ARatmospy

from ..miscellaneous import error_handling as errh
from .. import LOGGER


def create_tec_screen(fitsfile: pathlib.Path, fov: float,
                      bmax: float, t_int: int, duration: int,
                      ranseed: Union[None, int] = None):
    """
    Create Total Electron count (TEC) screen as a 4-D .fits image (x, y, time,
    frequency)

    Parameters
    ----------
    fitsfile
        Full path to write TEC screen to
    fov
        Field of view the TEC screen should cover [deg]
    bmax
        Maximum baseline length [m]
    t_int
        Time between consecutive phase screens [s]
    duration
        Total duration of the phase screen [s]
    ranseed
        Random number generator seed to use (optional)

    Returns
    -------

    """
    # Parameters that should come from args/kwargs
    screen_width_metres = 2 * 310e3 * np.tan(np.radians((fov / 2.)))
    rate = t_int / 60.0
    num_times = int(duration / t_int)

    # Parameters that should come from advanced settings/configuration
    r0 = 1e4  # Scale size [m] (section 5.1 of de Gasperin et al. 2018)
    sampling = 100.0  # Pixel-width [m]
    speed = 150e3 / 3600.0  # [m/s]
    alpha_mag = 0.999  # Evolve screen slowly  ## an arg/kwarg
    # Parameters for each layer.
    # (scale size [m], speed [m/s], direction [deg], layer height [m]).
    layer_params = np.array([(r0, speed, 60.0, 300e3),
                             (r0, speed / 2.0, -30.0, 310e3)])

    # Round screen width to nearest sensible number
    screen_width_metres = screen_width_metres / sampling // 100 * 100 * sampling

    m = int(bmax / sampling)  # Pixels per sub-aperture
    n = int(screen_width_metres / bmax)  # Sub-apertures across the screen
    num_pix = n * m
    pscale = screen_width_metres / (n * m)  # Pixel scale (100 m/pixel).
    print("Number of pixels %d, pixel size %.3f m" % (num_pix, pscale))
    print("Field of view %.1f (m)" % (num_pix * pscale))
    print('beginning ARatmos')
    tec_screen = ArScreens(n, m, pscale, rate, layer_params, alpha_mag, ranseed)
    tec_screen.run(num_times)
    print('ended ARatmos')

    # Convert to TEC
    # phase = image[pixel] * -8.44797245e9 / frequency
    phase2tec = -1e8 / 8.44797245e9

    hdr = fits.Header()
    hdr.set('SIMPLE', True)
    hdr.set('BITPIX', -32)
    hdr.set('NAXIS', 4)
    hdr.set('NAXIS1', num_pix)
    hdr.set('NAXIS2', num_pix)
    hdr.set('NAXIS3', num_times)
    hdr.set('NAXIS4', 1)
    # hdr.set('WCSAXES', 4)
    hdr.set('CTYPE1', 'XX')
    hdr.set('CTYPE2', 'YY')
    hdr.set('CTYPE3', 'TIME')
    hdr.set('CTYPE4', 'FREQ')
    hdr.set('CRVAL1', 0.0)
    hdr.set('CRVAL2', 0.0)
    hdr.set('CRVAL3', 0.0)
    hdr.set('CRVAL4', 1e8)
    hdr.set('CRPIX1', num_pix // 2 + 1)
    hdr.set('CRPIX2', num_pix // 2 + 1)
    hdr.set('CRPIX3', num_times // 2 + 1)
    hdr.set('CRPIX4', 1.0)
    hdr.set('CDELT1', pscale)
    hdr.set('CDELT2', pscale)
    hdr.set('CDELT3', 1.0 / rate)
    hdr.set('CDELT4', 1.0)
    hdr.set('CUNIT1', 'm')
    hdr.set('CUNIT2', 'm')
    hdr.set('CUNIT3', 's')
    hdr.set('CUNIT4', 'Hz')
    hdr.set('LATPOLE', 90.0)
    hdr.set('MJDREF', 0.0)

    while len(hdr) % 36 != 35:
        hdr.append()  # Adds a blank card to the end

    hdr.tofile(fitsfile, overwrite=True)

    shape = tuple(hdr[f'NAXIS{ii}'] for ii in range(1, hdr['NAXIS'] + 1))
    with open(fitsfile, 'rb+') as fobj:
        fobj.seek(len(hdr.tostring()) +
                  (np.product(shape) * np.abs(hdr['BITPIX'] // 8)) -
                  1)  # + 240 * 8)
        fobj.write(b'\0')

    with fits.open(fitsfile, memmap=True, mode='update') as hdul:
        for layer in range(len(tec_screen.screens)):
            for i, screen in enumerate(tec_screen.screens[layer]):
                hdul[0].data[:, i, ...] += phase2tec * screen[np.newaxis, ...]


def create_tec_screens(farm_cfg: 'FarmConfiguration',
                       scans: Tuple[Tuple[Time, Time], ...],
                       tec_prefix: str,
                       logger: Optional[logging.Logger] = None
                       ) -> List[pathlib.Path]:
    """
    Calculates and plots scan times for a farm configuration. Also optionally
    logs results/operations

    Parameters
    ----------
    farm_cfg
        FarmConfiguration instance to parse information from
    scans
        Tuple of (start, end)
    tec_prefix
        Prefix to append to ionospheric TEC .fits files
    logger
        logging.Logger instance to log messages to

    Returns
    -------
    None
    """
    from tqdm import tqdm

    if logger:
        logger.info(
            f"Creating TEC screens from scratch for {len(scans)} scans"
        )

    tec_root = farm_cfg.output_dcy / tec_prefix
    created_tec_fitsfiles = []
    for i, (t_start, t_end) in tqdm(enumerate(scans), desc='Creating TEC'):
        duration = (t_end - t_start).to_value('s')
        tec_fitsfile = tec_root.append(
            str(i).zfill(len(str(len(scans)))) + '.fits'
        )
        create_tec_screen(
            tec_fitsfile, 20., 20e3,
            farm_cfg.correlator.t_int, duration, farm_cfg.calibration.noise.seed
        )
        created_tec_fitsfiles.append(tec_fitsfile)

    if logger:
        logger.info(
            f"TEC screens saved to "
            f"{','.join([_.name for _ in created_tec_fitsfiles])}"
        )

    return created_tec_fitsfiles


class ArScreens(ARatmospy.ArScreens.ArScreens):
    """
    Adaptation of ARatmospy.ArScreens.ArScreens class. Purpose of this
    adaptation is to retain memory of the parameters used to instantiate and
    ArScreens instance for future use. After instantiation of an ArScreens
    instance, parameters can not be changed.

    ArScreens is used to generate atmosphere phase screens using an
    autoregressive process to add stochastic noise to an otherwise frozen flow.
    The adaptation is to ensure that the input parameters to the class's
    constructor are saved as attributes for future use.
    """
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
        super().__init__(n, m, pscale, rate, paramcube, alpha_mag, ranseed)
        self._n = n
        self._m = m
        self._pscale = pscale
        self._rate = rate
        self._paramcube = paramcube
        self._alpha_mag = alpha_mag
        self._ranseed = ranseed

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
    def paramcube(self):
        """
        Parameter array describing each layer of the atmosphere to be
        modeled.  Each row contains a tuple of (r0 [m], velocity [m/s],
        direction [deg], altitude [m]) describing the corresponding layer
        """
        return self._paramcube

    @property
    def alpha_mag(self):
        """
        Magnitude of autoregressive parameter, whereby (1-alpha_mag) is the
        fraction of the phase from the prior time step that is "forgotten" and
        replaced by Gaussian noise
        """
        return self._alpha_mag

    @property
    def ranseed(self):
        """Random number generator seed"""
        return self._ranseed


class TECScreen:
    """
    Class to hold all attributes/methods concerning a complete TEC screen
    generation
    """
    @classmethod
    def create_from_fits(cls, fitsfile: pathlib.Path,
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
        TECScreen instance. Be aware there will be no ArScreens attribute on the
        created TECScreen instance
        """
        with fits.open(fitsfile, memmap=True) as hdul:
            hdr = hdul[0].header
            duration = hdr['CDELT3'] * u.s * hdr['NAXIS3']
            t_end = t_start + duration

        tecscreen = cls(arscreen=None, t_start=t_start, t_end=t_end)
        tecscreen._cube = fitsfile

        return tecscreen

    def __init__(self, arscreen: ArScreens, t_start: Time, t_end: Time):
        """
        Parameters
        ----------
        arscreen
            ARatmospy.ArScreens.ArScreens instance
        t_start
            Start time of the TEC screen
        t_end
            End time of the TEC screen
        """
        self._arscreen = arscreen
        self._t_start = t_start
        self._t_end = t_end
        self._times: Tuple = None
        self._cube: pathlib.Path = None

    @property
    def arscreen(self) -> ArScreens:
        """farm.calibration.tec.ArScreens instance"""
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
        """Full path to .fits cube produced by ArScreens instance"""
        return self._cube

    def create_tec_screen(self, fitsfile: pathlib.Path):
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

        # Convert to TEC
        # phase = image[pixel] * -8.44797245e9 / frequency
        freq = 1e8
        phase2tec = -freq / 8.44797245e9

        hdr = fits.Header()
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

        while len(hdr) % 36 != 35:
            hdr.append()  # Adds a blank card to the end

        hdr.tofile(fitsfile, overwrite=True)

        shape = tuple(hdr[f'NAXIS{ii}'] for ii in range(1, hdr['NAXIS'] + 1))
        with open(fitsfile, 'rb+') as fobj:
            fobj.seek(len(hdr.tostring()) +
                      (np.product(shape) * np.abs(hdr['BITPIX'] // 8)) -
                      1)
            fobj.write(b'\0')

        with fits.open(fitsfile, memmap=True, mode='update') as hdul:
            for layer in range(len(self.arscreen.screens)):
                for i, screen in enumerate(self.arscreen.screens[layer]):
                    hdul[0].data[:, i, ...] += phase2tec * screen[np.newaxis, ...]

        if not fitsfile.exists():
            errh.raise_error(FileNotFoundError,
                             f'TEC Screen not generated ({fitsfile} not '
                             f'written)')

        self._cube = fitsfile.resolve()

    def extract_tec_screen_slice(self, slice_fitsfile: pathlib.Path,
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

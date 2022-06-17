import pathlib
import logging
from typing import Union, Optional, List, Tuple
import numpy as np
from astropy.io import fits
from astropy.time import Time
from ARatmospy.ArScreens import ArScreens

from ..data.loader import FarmConfiguration


def create_tec_screen(fitsfile: pathlib.Path, freq: float, fov: float,
                      bmax: float, t_int: int, duration: int,
                      ranseed: Union[None, int] = None):
    # Parameters that should come from args/kwargs
    # freq = 1.4e8   # should be mean frequency of bandwidth [Hz]
    screen_width_metres = 2 * 310e3 * np.tan(np.radians((fov / 2.)))
    # bmax = 20e3  # sub-aperture size i.e. maximum baseline length [m]
    # rate = 1.0 / 60.0   # [Hz]  ## visibility integration time?
    rate = t_int / 60.0
    # num_times = 4 * 60  # == four hours
    num_times = int(duration / t_int)

    # Parameters that should come from advanced settings/configuration
    r0 = 5e3  # Scale size [m]
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
    tec_screen = ArScreens(n, m, pscale, rate, layer_params, alpha_mag)
    tec_screen.run(num_times)
    print('ended ARatmos')

    # Convert to TEC
    # phase = image[pixel] * -8.44797245e9 / frequency
    phase2tec = -freq / 8.44797245e9

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
    hdr.set('CRVAL4', freq)
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
                  1)# + 240 * 8)
        fobj.write(b'\0')

    with fits.open(fitsfile, memmap=True, mode='update') as hdul:
        for layer in range(len(tec_screen.screens)):
            for i, screen in enumerate(tec_screen.screens[layer]):
                hdul[0].data[:, i, ...] += phase2tec * screen[np.newaxis, ...]


def create_tec_screens(farm_cfg: FarmConfiguration,
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
    scan_times
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
            tec_fitsfile, np.mean(farm_cfg.correlator.frequencies), 20., 20e3,
            farm_cfg.correlator.t_int, duration, farm_cfg.calibration.noise.seed
        )
        created_tec_fitsfiles.append(tec_fitsfile)

    if logger:
        logger.info(
            f"TEC screens saved to "
            f"{','.join([_.name for _ in created_tec_fitsfiles])}"
        )

    return created_tec_fitsfiles


if __name__ == '__main__':
    fitsfile = pathlib.Path('/Users/simon.purser/Desktop/test_TEC.fits')
    create_tec_screen(fitsfile, [1e8])

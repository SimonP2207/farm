#!/usr/bin/env python3
"""
Pipeline for the creation of a synthetic dataset, and its corresponding
deconvolved image, containing the epoch of reionisation's hydrogen 21cm signal
"""
import os
import logging
import toml
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
import py21cmfast as p21c
import tools21cm as t2c


if __name__ == '__main__':
    from farm.miscellaneous.plotting import plot_lightcone as plot_lc
    from farm import LOG_FMT, LOG_DATE_FMT
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
else:
    from ..miscellaneous.plotting import plot_lightcone as plot_lc
    from .. import LOGGER, LOG_FMT, LOG_DATE_FMT


def z_from_freq(nu):
    """Observed frequency [Hz] to redshift for Hydrogen 21cm line"""
    nu0 = 1420.40575e6

    return (nu0 / nu) - 1.


def create_eor_h21cm_fits(params_file: str):
    """
    Create the .fits cube of the Epoch of Reionisation's hydrogen-21cm signal

    Parameters
    ----------
    params_file
        Full path to .toml configuration file for producing the .fits cube
    """
    from datetime import datetime

    with open(params_file, 'rt') as f:
        params = toml.load(f)

    output_dir = params['output']['output_dcy'].rstrip('/')
    # ######################################################################## #
    # ######################## Set up the logger ############################# #
    # ######################################################################## #
    pipeline_start = datetime.now()
    logfile_name = f'EoR_H21cm_{pipeline_start.strftime("%Y%b%d_%H%M%S").upper()}.log'
    logfile = f"{output_dir}{os.sep}{logfile_name}"
    LOGGER.setLevel(logging.DEBUG)

    # Set up handler for writing log messages to log file
    file_handler = logging.FileHandler(
        str(logfile), mode="w", encoding=sys.stdout.encoding
    )
    log_formatter = logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # Set up handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(LOG_LEVEL)
    LOGGER.addHandler(console_handler)
    # ######################################################################## #
    # ###################### Parse configuration from file ################### #
    # ######################################################################## #
    output_fits = f"{output_dir}{os.sep}{params['output']['root_name']}.fits"
    random_seed = params['user_params']['seed']
    ra_deg = params['field']['ra0']  # [deg]
    dec_deg = params['field']['dec0']  # [deg]
    fov_deg = params['field']['fov']  # [deg]
    freqs = np.linspace(params['correlator']['freq_min'],
                        params['correlator']['freq_max'],
                        params['correlator']['nchan'])
    n_output_cell = params['field']['n_cells']
    plot_light_cone = params['user_params']['plot_lc']

    # Number of cells per side for the low res box (output cube)
    HII_DIM = n_output_cell
    # Number of cells for the high res box (sampling initial conditions).
    # Should be at least 3 times HII_DIM.
    DIM = 3 * HII_DIM
    astro_params = params['astro_params']
    dfreq = np.ptp(freqs[:2])
    redshift = z_from_freq(freqs).tolist()
    zmin = np.amin(redshift)
    zmax = np.amax(redshift)

    # Desired field of view at highest redshift
    fov_mpc = Planck18.comoving_transverse_distance(zmax).value * \
              np.deg2rad(fov_deg)

    # Length of the box in Mpc (simulation size) i.e. the comoving size
    BOX_LEN = fov_mpc

    flag_options = params['flags']

    user_params : dict = params["user_params"]
    user_params['N_THREADS'] = params['user_params']['n_cpu']
    user_params["HII_DIM"]= HII_DIM
    user_params["BOX_LEN"] = BOX_LEN
    user_params["DIM"] = DIM

    user_params.pop('n_cpu')
    user_params.pop('seed')
    user_params.pop('plot_lc')

    if flag_options['USE_MINI_HALOS']:
        user_params['USE_RELATIVE_VELOCITIES'] = True

    # ######################################################################## #
    # ###################### Create relevant directories ##################### #
    # ######################################################################## #
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cache_dir = f"{output_dir}{os.sep}_cache"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    p21c.config['direc'] = cache_dir

    # clear the cache so that we get the same result each time
    LOGGER.info("Clearing the cache")
    p21c.cache_tools.clear_cache(direc=cache_dir)
    # ######################################################################## #
    # ###################### Parse configuration from file ################### #
    # ######################################################################## #
    LOGGER.info("Computing initial conditions")
    initial_conditions = p21c.initial_conditions(
        user_params=user_params, random_seed=random_seed, direc=output_dir,
    )

    lightcone_quantities = ('brightness_temp',)
    LOGGER.info("Generating lightcone")
    lightcone = p21c.run_lightcone(
        redshift=zmin,
        # max_redshift = 10.0,
        init_box=initial_conditions,
        flag_options=flag_options,
        astro_params=astro_params,
        lightcone_quantities=lightcone_quantities,
        global_quantities=lightcone_quantities,
        random_seed=random_seed,
        direc=output_dir,
    )

    lc0 = getattr(lightcone, 'brightness_temp')
    zs0 = lightcone.lightcone_redshifts

    # so cut a lightcone in a redshift (=freq) range I want
    z_start_index = min(range(len(zs0)), key=lambda i: abs(zs0[i] - zmin))
    z_end_index = min(range(len(zs0)), key=lambda i: abs(zs0[i] - zmax))

    zs = zs0[z_start_index:z_end_index]
    lc = lc0[:, :, z_start_index:z_end_index]

    # plotting physical lightcone
    if plot_light_cone:
        output_png = output_fits.replace('.fits', '.png')
        LOGGER.info(f"Plotting lightcone and saving to {output_png}")
        try:
            plot_lc(lc, zs, fov=BOX_LEN, title='Physical lightcone', xlabel='z',
                    ylabel='L (cMpc)', savefig=output_png)
        except Exception as e:
            LOGGER.error(f"{e.__class__.__name__}: {e}")
            LOGGER.error("Could not plot lightcone")
    # converting physical to observational coordinates - given cosmology is
    # different here
    angular_size_deg = t2c.angular_size_comoving(BOX_LEN, zs)
    LOGGER.info('Minimum angular size: {:.2f} degrees'.format(angular_size_deg.min()))
    LOGGER.info('Maximum angular size: {:.2f} degrees'.format(angular_size_deg.max()))

    physical_freq = t2c.z_to_nu(zs)  # redshift to frequencies in MHz

    LOGGER.info(
        'Minimum frequency gap in the physical light-cone data: '
        '{:.2f} MHz'.format(np.abs(np.gradient(physical_freq)).min())
    )
    LOGGER.info(
        'Maximum frequency gap in the physical light-cone data: '
        '{:.2f} MHz'.format(np.abs(np.gradient(physical_freq)).max())
    )

    zmin = zs.min()
    zmax = zs.max()
    output_dtheta = (fov_deg / (n_output_cell + 1)) * 60  # [arcmin]
    # Original physical_lightcone_to_observational padded highest redshifts by
    # repeating flux distribution. AB + EL didn't like that, and changed it to
    # trim whilst starting with a bigger cube.
    LOGGER.info("Converting physical lightcone to observational")
    obs_lc, obs_freq = t2c.physical_lightcone_to_observational(
        lc, zmin, zmax, dfreq / 1e6, output_dtheta, input_box_size_mpc=BOX_LEN
    )  # Mpc * Mpc * Mpc to deg * deg * Hz

    # save to a fits file
    lc_out = np.float32(obs_lc.transpose())
    lc_out /= 1000.  # mK to K

    hdu = fits.PrimaryHDU(lc_out)
    hdul = fits.HDUList([hdu])

    hdul[0].header.set('CTYPE1', 'RA---SIN')
    hdul[0].header.set('CTYPE2', 'DEC--SIN')
    hdul[0].header.set('CTYPE3', 'FREQ    ')
    hdul[0].header.set('CRVAL1', ra_deg)
    hdul[0].header.set('CRVAL2', dec_deg)
    hdul[0].header.set('CRVAL3', np.min(freqs))
    hdul[0].header.set('CRPIX1', -HII_DIM / 2.)
    hdul[0].header.set('CRPIX2', HII_DIM / 2.)
    hdul[0].header.set('CRPIX3', 1)
    hdul[0].header.set('CDELT1', fov_deg / HII_DIM)
    hdul[0].header.set('CDELT2', fov_deg / HII_DIM)
    hdul[0].header.set('CDELT3', dfreq)
    hdul[0].header.set('CUNIT1', 'deg     ')
    hdul[0].header.set('CUNIT2', 'deg     ')
    hdul[0].header.set('CUNIT3', 'Hz      ')
    hdul[0].header.set('BUNIT', 'K       ')

    LOGGER.info(f"Writing output lightcone to {output_fits}")
    hdul.writeto(output_fits, overwrite=True)


if __name__ == '__main__':
    import sys
    import argparse
    import pathlib
    # ######################################################################## #
    # ########## Parse configuration from file or from command-line ########## #
    # ######################################################################## #
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("params_file",
                            help="Full path to EoR H21cm configuration .toml "
                                 "file",
                            type=str)
        parser.add_argument("-d", "--debug",
                            help="Set terminal log output to verbose levels",
                            action="store_true")
        args = parser.parse_args()
        params_file = pathlib.Path(args.params_file)
        LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
    else:
        params_file = pathlib.Path('/data-cold/for-backup/sdc/SDC3/foregrounds/EoR_lightcones/v4/EoR_H21cm_v4.toml')
        LOG_LEVEL = logging.DEBUG

    create_eor_h21cm_fits(params_file)

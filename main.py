#!/usr/bin/env python3
import os
import shutil
import sys
import subprocess
import logging
import argparse
import pathlib
from typing import ByteString, Union, Tuple, List, Optional
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.time import Time

import farm
import farm.data.loader as loader
import farm.physics.astronomy as ast
import farm.miscellaneous as misc
from farm.miscellaneous import generate_random_chars as grc
import farm.miscellaneous.error_handling as errh
from farm.miscellaneous.image_functions import regrid_fits, pb_multiply
import farm.miscellaneous.plotting as plotting
from farm.observing import Scan, Observation
import farm.sky_model.tb_functions as tb_funcs
from farm.software import casa
from farm.software.miriad import miriad
from farm.software import oskar
from farm.software.oskar import (run_oskar_sim_beam_pattern,
                                 run_oskar_sim_interferometer)
from farm import LOGGER

from dataclasses import dataclass, field
from collections.abc import Iterable
import numpy.typing as npt


def is_miriad_vis_file(filename: pathlib.Path) -> bool:
    """
    

    Parameters
    ----------
    filename : pathlib.Path
        Path to prospective miriad visibility file

    Returns
    -------
    bool
        Whether filename is a miriad visibility file (True) 
        or not (False)
    """
    if not filename.is_dir():
        return False

    req_contents = ['flags', 'header', 'history', 'vartable', 'visdata']
    contents = list(filename.iterdir())

    return all([filename / f in contents for f in req_contents])


def write_mir_gains_table(mirfile: pathlib.Path, gheader: ByteString,
                          gtimes: npt.NDArray, ggains: npt.NDArray):
    """
    Write a set of gains to the gains table of a miriad visibility
    data file

    Parameters
    ----------
    mirfile : pathlib.Path
        Full path to miriad visibility data file
    gheader : ByteString
        Gains header
    gtimes : npt.NDArray
        Array of times corresponding to each gains interval
    ggains : npt.NDArray
        Array of gains of shape (number of antennae, 2)

    Returns
    -------
    None
    """
    from struct import pack

    n = len(gtimes)
    ngains = ggains.shape[1]
    with open(mirfile / 'gains', 'wb') as f:
        f.write(gheader)
        for i in range(n):
            f.write(pack('>d', gtimes[i]))
            f.write(pack(f'>{ngains * 2:.0f}f',
                         *ggains[i, :, :].flatten().tolist()))


def read_mir_gains_table(mirfile: pathlib.Path) -> Tuple[
    ByteString, npt.NDArray, npt.NDArray]:
    """
    Read the gains table from a miriad visibility data file and return
    its header, times and gains

    Parameters
    ----------
    mirfile : pathlib.Path
        Full path to miriad visibility data file

    Returns
    -------
    Tuple[ByteString, npt.NDArray, npt.NDArray]
        Tuple of gains header, times and gain values. The latter is of
        shape (number of antennae, 2)
    
    Raises
    ------
    ValueError
        If mirfile is not a miriad visibility data file, or it is but 
        does not contain a gains table
    """
    from struct import unpack

    if not is_miriad_vis_file(mirfile):
        err_msg = f'{mirfile} is not miriad visibility data'
        errh.raise_error(ValueError, err_msg)

    if not (mirfile / 'gains').exists():
        err_msg = f'{mirfile} does not contain a gains table'
        errh.raise_error(ValueError, err_msg)

    # read header items we need
    ngains, nfeed, ntau, nsols = 0, 0, 0, 0
    items = [b'ngains', b'nfeeds', b'ntau', b'nsols']

    # Parse necessary information from the header
    with open(mirfile / 'header', 'rb') as f:
        line = f.read(16)
        while line:
            ln = unpack('!16B', line)[15]
            # round up to multiple of 16
            ln = 16 * ((ln + 15) // 16)
            item = unpack('!15s', line[0:15])[0].split(b'\0')[0]
            data = f.read(ln)

            if item in items:
                val = unpack('!i', data[4:8])[0]
                if item == b'nfeeds':
                    nfeeds = val
                if item == b'ngains':
                    ngains = val
                if item == b'ntau':
                    ntau = val
                if item == b'nsols':
                    nsols = val
            line = f.read(16)

    n = max(1, nsols)
    with open(mirfile / 'gains', 'rb') as f:
        gtimes = np.zeros(n)
        ggains = np.zeros((n, ngains, 2), dtype=np.float32)
        gheader = f.read(8)
        for i in range(n):
            buf = f.read(8)
            time = unpack('>d', buf)
            gtimes[i] = time[0]
            buf = f.read(ngains * 8)
            g = unpack(f'>{ngains * 2:.0f}f', buf)
            ggains[i, :, :] = np.array(g).reshape(ngains, 2)

    return gheader, gtimes, ggains


def implement_gain_errors(vis_file: pathlib.Path, t_interval: float,
                          pnoise: float, gnoise: float, rseed: int):
    """
    Introduce gains errors in to a miriad visibility data file

    Parameters
    ----------
    vis_file : pathlib.Path
        Full path to miriad visibility data file
    t_interval : float
        Interval between gain solutions [minutes]
    pnoise : float
        Phase error [deg]
    gnoise : float
        Amplitude error [percentage]
    rseed : int
        Random number generator seed
    """
    from numpy.random import default_rng

    if t_interval <= 60.:
        err_msg = "t_interval must be longer than 1h or get BP problem"
        errh.raise_error(ValueError, err_msg)

    if not is_miriad_vis_file(vis_file):
        err_msg = f'{vis_file} is not miriad visibility data'
        errh.raise_error(ValueError, err_msg)

    if not (vis_file / 'gains').exists():
        # First run gperror to make a gain table with some nominal values
        # (since random number seed can not be passed)
        LOGGER.info(f"Creating gains table in {vis_file}")
        miriad.gperror(vis=vis_file, interval=t_interval,
                       pnoise=pnoise, gnoise=gnoise)

    # Now read the gain table and replace with some nice random numbers
    gheader, gtimes, ggains = read_mir_gains_table(vis_file)

    phas_rms = pnoise * np.pi / 180.
    gain_rms = gnoise / 100.
    rng = default_rng(seed=rseed)
    gvals = rng.normal(loc=1., scale=gain_rms,
                       size=ggains[:, :, 0].shape)
    pvals = rng.normal(loc=0., scale=phas_rms,
                       size=ggains[:, :, 1].shape)
    cvals = (np.cos(pvals) + 1j * np.sin(pvals)) * gvals
    rvals = cvals.real.astype('float32')
    ivals = cvals.imag.astype('float32')
    my_ggains = np.stack((rvals, ivals), axis=2)
    write_mir_gains_table(vis_file, gheader, gtimes, my_ggains)


def write_mir_bandpass_table(mirfile: pathlib.Path, bheader: ByteString,
                             btimes: npt.NDArray, bgains: npt.NDArray):
    """
    Write a set of gains to the bandpass table of a miriad visibility
    data file

    Parameters
    ----------
    mirfile : pathlib.Path
        Full path to miriad visibility data file
    bheader : ByteString
        Gains header
    btimes : npt.NDArray
        Array of times corresponding to bandpass solution interval
    bgains : npt.NDArray
        Array of bandpass solutions of shape (number of antennae,
        number of antennae, number of channels, 2)

    Returns
    -------
    None
    """
    from struct import pack

    n = len(btimes)
    nbpsols = n
    if n == 1 and btimes[0] == 0:
        nbpsols = 0

    ngains, nchan = bgains.shape[1:3]
    with open(mirfile / 'bandpass', 'wb') as f:
        f.write(bheader)
        for i in range(n):
            f.write(pack(f'>{ngains * nchan * 2:.0f}f',
                         *bgains[i, :, :, :].flatten().tolist()))
            if nbpsols > 0:
                f.write(pack('>d', btimes[i]))


def write_mir_freqs_table(mirfile: pathlib.Path, fheader: ByteString,
                          nchan: int, freq0: float, chan_width: float):
    from struct import pack

    with open(mirfile / 'freqs', 'wb') as f:
        f.write(fheader)
        f.write(pack('>iidd', nchan, 0, freq0 / 1e9, chan_width / 1e9))


def implement_bandpass_errors(vis_file: pathlib.Path, nchan: int,
                              freq0: float, chan_width: float,
                              pnoise: float, gnoise: float, rseed: int):
    """
    Introduce bandpass errors in to a miriad visibility data file

    Parameters
    ----------
    vis_file : pathlib.Path
        Full path to miriad visibility data file
    nchan : int
        Number of channels in data file
    pnoise : float
        Phase error [deg]
    gnoise : float
        Amplitude error [percentage]
    rseed : int
        Random number generator seed
    """
    from numpy.random import default_rng

    rng = default_rng(seed=rseed)

    gheader, gtimes, ggains = read_mir_gains_table(vis_file)

    bgains = np.zeros((ggains.shape[0], ggains.shape[1], nchan, 2),
                      dtype='float32')
    gvals = rng.normal(loc=1., scale=gnoise, size=bgains[:, :, :, 0].shape)
    pvals = rng.normal(loc=0., scale=pnoise, size=bgains[:, :, :, 1].shape)
    cvals: npt.NDArray = (np.cos(pvals) + 1j * np.sin(pvals)) * gvals
    rvals = cvals.real.astype('float32')
    ivals = cvals.imag.astype('float32')

    my_bgains = np.stack((rvals, ivals), axis=3)
    write_mir_bandpass_table(vis_file, gheader, gtimes, my_bgains)
    write_mir_freqs_table(vis_file, gheader, nchan, freq0, chan_width)

    miriad.puthd(_in=f'{str(vis_file)}/nbpsols', value=len(gtimes))
    miriad.puthd(_in=f'{str(vis_file)}/nspect0', value=1.)
    miriad.puthd(_in=f'{str(vis_file)}/nchan0', value=nchan)


def create_beam_pattern_fits(cfg, scan_num, time, dt, beam_fits):
    beam_sfx = '_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I'
    beam_root = cfg.root_name.append(f"_scan{scan_num}")
    beam_name = beam_root.append(beam_sfx)
    beam_fname = beam_name.append('.fits')

    # Create beam pattern as .fits cube using oskar's sim_beam_pattern task
    LOGGER.info(f"Running oskar_sim_beam_pattern from {cfg.sbeam_ini}")
    cfg.set_oskar_sim_beam_pattern("beam_pattern/root_path", beam_root)
    cfg.set_oskar_sim_beam_pattern("observation/start_time_utc", time)
    cfg.set_oskar_sim_beam_pattern("observation/length", dt)
    run_oskar_sim_beam_pattern(cfg.sbeam_ini)

    beam_hdu = fits.open(beam_fname)

    LOGGER.info(f"Starting synthetic observations' scan #{scan_num}")
    # TODO: End of 27APR22. Figure out sbmout etc below. I think we don't
    #  need a lot of the files as each scan's beam has only one 'frame' in
    #  time in the beam cube
    bmout_hdu = fits.PrimaryHDU(beam_hdu[0].data[0, :, :, :])
    bmout_hdu.header.set('CTYPE1', 'RA---SIN')
    bmout_hdu.header.set('CTYPE2', 'DEC--SIN')
    bmout_hdu.header.set('CTYPE3', 'FREQ    ')
    bmout_hdu.header.set('CRVAL1', cfg.field.coord0.ra.deg)
    bmout_hdu.header.set('CRVAL2', cfg.field.coord0.dec.deg)
    bmout_hdu.header.set('CRVAL3', cfg.correlator.freq_min)
    bmout_hdu.header.set('CRPIX1', cfg.field.nx // 2)
    bmout_hdu.header.set('CRPIX2', cfg.field.ny // 2)
    bmout_hdu.header.set('CRPIX3', 1)
    bmout_hdu.header.set('CDELT1', -cfg.field.cdelt)
    bmout_hdu.header.set('CDELT2', cfg.field.cdelt)
    bmout_hdu.header.set('CDELT3', cfg.correlator.freq_inc)
    bmout_hdu.header.set('CUNIT1', 'deg     ')
    bmout_hdu.header.set('CUNIT2', 'deg     ')
    bmout_hdu.header.set('CUNIT3', 'Hz      ')
    bmout_hdu.writeto(beam_fits, overwrite=True)


def determine_visfile_type(visfile: Union[str, pathlib.Path]) -> str:
    """
    Determine the type of visbility data file. One of either
    'uvfits', 'miriad', or 'ms' for uvfits format, miriad visibility
    format, or measurement set, respectively
    """
    if isinstance(visfile, str):
        visfile = pathlib.Path(visfile)

    try:
        fits.open(visfile)
        return 'uvfits'
    except IsADirectoryError:
        pass
    except OSError:
        return ''

    visfile_contents = os.listdir(visfile)

    if 'visdata' in visfile_contents:
        return 'miriad'

    if 'ANTENNA' in visfile_contents:
        return 'ms'

    return ''


# TODO: Write check_tec_image_compatibility method
def check_tec_image_compatibility(farm_cfg: loader.FarmConfiguration,
                                  tec_images: List[pathlib.Path]
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


# ############################################################################ #
# ############ Parse configuration from file or from command-line ############ #
# ############################################################################ #
if len(sys.argv) != 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        help="Full path to farm configuration .toml file",
                        type=str)
    parser.add_argument("-m", "--model-only",
                        help="Compute and output sky model only",
                        action="store_true")
    parser.add_argument("-d", "--debug",
                        help="Set terminal log output to verbose levels",
                        action="store_true")
    args = parser.parse_args()
    config_file = pathlib.Path(args.config_file)
    model_only = args.model_only
    log_level = logging.DEBUG if args.debug else logging.INFO
else:
    config_file = pathlib.Path(farm.data.FILES['EXAMPLE_CONFIG'])
    model_only = False
    log_level = logging.DEBUG

cfg = loader.FarmConfiguration(config_file)
if len(sys.argv) != 1:
    os.chdir(cfg.output_dcy)

dryrun = False
# ############################################################################ #
# ######################## Set up the logger ################################# #
# ############################################################################ #
now = datetime.now()
logfile = cfg.output_dcy / f'farm{now.strftime("%Y%b%d_%H%M%S").upper()}.log'
LOGGER.setLevel(logging.DEBUG)

# Set up handler for writing log messages to log file
file_handler = logging.FileHandler(
    str(logfile), mode="w", encoding=sys.stdout.encoding
)
log_formatter = logging.Formatter(farm.LOG_FMT, datefmt=farm.LOG_DATE_FMT)
file_handler.setFormatter(log_formatter)
LOGGER.addHandler(file_handler)

# Set up handler to print to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(log_level)
LOGGER.addHandler(console_handler)
# ############################################################################ #
# ########### Create run's root directory if it doesn't exists ############### #
# ############################################################################ #
if not cfg.output_dcy.exists():
    LOGGER.info(f"Creating output directory, {cfg.output_dcy}")
    cfg.output_dcy.mkdir(exist_ok=True)
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Initial set up of OSKAR ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
sky_fov, sky_side_lobes = None, None
if not model_only:
    LOGGER.info("Setting up Oskar Sky instances")
    sky_fov = oskar.Sky()
    sky_side_lobes = oskar.Sky()
# ############################################################################ #
# ############################################################################ #
# #####################  # # # # # # # # # # # # # # # ####################### #
# ########################### SkyModel Creation ############################## #
# #####################  # # # # # # # # # # # # # # # ####################### #
# ############################################################################ #
LOGGER.info(f"Instantiating SkyModel")
sky_model = farm.sky_model.SkyModel((cfg.field.nx, cfg.field.ny),
                                    cfg.field.cdelt, cfg.field.coord0,
                                    cfg.correlator.frequencies)
# ############################################################################ #
# ################## Large-scale Galactic foreground model ################### #
# ############################################################################ #
gdsm = None
if cfg.sky_model.galactic.large_scale_component.include:
    if cfg.sky_model.galactic.large_scale_component.create:
        LOGGER.info("Creating large-scale Galactic foreground images")
        gdsm = farm.sky_model.SkyComponent(
            name='GDSM', npix=(cfg.field.nx, cfg.field.ny),
            cdelt=cfg.field.cdelt, coord0=cfg.field.coord0,
            tb_func=tb_funcs.gdsm2016_t_b
        )
        gdsm.add_frequency(cfg.correlator.frequencies)
    else:
        fits_gdsm = cfg.sky_model.galactic.small_scale_component.image
        if not fits_gdsm.exists():
            errh.raise_error(FileNotFoundError, f"{fits_gdsm} does not exist")
        elif fits_gdsm.is_dir():
            errh.raise_error(FileNotFoundError, f"{fits_gdsm} is a directory")

        LOGGER.info(f"Loading Galactic large-scale component from {fits_gdsm}")
        gdsm = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_gdsm, name='GDSM',
            freqs=cfg.correlator.frequencies
        )
        gdsm = gdsm.regrid(sky_model)
    # In the case that the small-scale Galactic emission is not included in
    # the SkyModel, add the large-scale emission component alone to the
    # SkyModel
    if not cfg.sky_model.galactic.small_scale_component:
        sky_model.add_component(gdsm)
    else:
        if not dryrun:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {gdsm.name} "
                        f"SkyComponent")
        else:
            gdsm.write_fits(cfg.output_dcy / f"{gdsm.name}_component.fits",
                            unit='JY/PIXEL')
else:
    LOGGER.info("Not including large-scale Galactic component")
 # ############################################################################ #
# ################ Small-scale Galactic foreground model ##################### #
# ############################################################################ #
gssm = None
if cfg.sky_model.galactic.small_scale_component.include:
    if cfg.sky_model.galactic.small_scale_component.create:
        errh.raise_error(NotImplementedError,
                         "Currently can only load GSSM model from fits image")
    else:
        fits_gssm = cfg.sky_model.galactic.small_scale_component.image
        if not fits_gssm.exists():
            errh.raise_error(FileNotFoundError, f"{fits_gssm} does not exist")
        elif fits_gssm.is_dir():
            errh.raise_error(FileNotFoundError, f"{fits_gssm} is a directory")

        LOGGER.info(f"Loading Galactic small-scale component from {fits_gssm}")
        with fits.open(fits_gssm) as hdulist:
            fits_nx = np.shape(hdulist[0].data)[-1]

        # cdelt/coord0 replace fits header values here since small-scale image
        # is not a real observation
        gssm = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_gssm,
            name='GSSM',
            cdelt=cfg.field.fov[0] / fits_nx,
            coord0=cfg.field.coord0,
            freqs=cfg.correlator.frequencies
        )

        rot_angle = ast.angle_to_galactic_plane(cfg.field.coord0)
        LOGGER.info(f"Rotating Galactic small-scale component by "
                    f"{np.degrees(rot_angle):.1f}deg")
        gssm.rotate(angle=rot_angle, inplace=True)

        LOGGER.info("Regridding small-scale image")
        gssm = gssm.regrid(sky_model)

    if cfg.sky_model.galactic.large_scale_component.include:
        LOGGER.info("Normalising small/large-scale power spectra")
        gssm.normalise(gdsm, inplace=True)
        # TODO: Hard-coded beam information here (56 arcmin)
        merged = gssm.merge(gdsm, (gssm.cdelt, gssm.cdelt, 0.),
                            (56. / 60., 56. / 60., 0.), 'GASM')
        sky_model.add_component(merged)
        if not dryrun:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {gssm.name} "
                        f"SkyComponent")
        else:
            gssm.write_fits(cfg.output_dcy / f"{gssm.name}_component.fits",
                            unit='JY/PIXEL')
    else:
        sky_model.add_component(gssm)
else:
    LOGGER.info("Not including small-scale Galactic component")
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ######################## Extragalactic foreground model #################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########################## Real sources from surveys ####################### #
# ############################################################################ #
# Also, separate sky model for in-fov and out-fov sources with flux cutoff
# for each
eg_real = None
if cfg.sky_model.extragalactic.real_component.include:
    LOGGER.info("Incorporating known point sources into foreground model")
    if cfg.sky_model.extragalactic.real_component.create:
        errh.raise_error(NotImplementedError,
                         "Currently can only load model from fits")
    else:
        fits_eg_real = cfg.sky_model.extragalactic.real_component.image

        if not fits_eg_real.exists():
            errh.raise_error(FileNotFoundError, f"{fits_eg_real} doesn't exist")
        elif fits_eg_real.is_dir():
            errh.raise_error(FileNotFoundError, f"{fits_eg_real} is a dir")

        data = None
        LOGGER.info("Loading catalogue")
        if misc.fits.is_fits(fits_eg_real):
            if misc.fits.is_fits_table(fits_eg_real):
                # Parse .fits table data
                data = misc.fits.fits_table_to_dataframe(fits_eg_real)
            else:
                errh.raise_error(NotImplementedError,
                                 "Can't load known points sources from .fits"
                                 "image")
        elif misc.file_handling.is_osm_table(fits_eg_real):
            data = misc.file_handling.osm_to_dataframe(fits_eg_real)

        else:
            errh.raise_error(Exception, f"Unsure of format of {fits_eg_real}")

        # Add needed columns to DataFrame
        # TODO: Put hard-coded, generic spectral-index of -0.7 somewhere
        #  sensible
        data.loc[np.isnan(data.spix), 'spix'] = -0.7

        # Create source masks for field of view and side lobes
        mask_fov = ast.within_square_fov(
            cfg.field.fov, cfg.field.coord0.ra.deg,
            cfg.field.coord0.dec.deg, data.ra, data.dec
        )

        cfg_eg_real = cfg.sky_model.extragalactic.real_component

        # Mask which excludes sources below outer flux lower-limit
        mask_side_lobes = data.fluxI > cfg_eg_real.flux_outer

        # Mask excluding known sources below the transition threshold
        flux_range_mask = (
                (data.fluxI < cfg_eg_real.flux_inner) &
                (data.fluxI > cfg_eg_real.flux_transition)
        )

        # Create SkyComponent instance and save .fits image of GLEAM
        # sources
        eg_real = farm.sky_model.SkyComponent.load_from_dataframe(
            data, 'EG_Known', cfg.field.cdelt,
            cfg.field.coord0, fov=cfg.field.fov,
            freqs=cfg.correlator.frequencies,
            flux_range=(cfg_eg_real.flux_transition,
                        cfg_eg_real.flux_inner),
            beam=None
        )

        if not dryrun:
            eg_real.write_fits(
                cfg.output_dcy / f"{eg_real.name}_component.fits",
                unit='JY/PIXEL'
            )
        else:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {eg_real.name} "
                        f"SkyComponent")

        # Below line commented out so that any Gaussian sources are added to
        # Oskar to deal with
        # sky_model.add_component(eg_real)

        if not model_only:
            # Added to SkyModel instead
            # Add to fov Sky instance
            oskar.add_dataframe_to_sky(data[mask_fov & flux_range_mask],
                                       sky_fov)

            # Add to side-lobes Sky instance
            oskar.add_dataframe_to_sky(data[mask_side_lobes], sky_side_lobes)

else:
    LOGGER.info("Not including known point sources into foreground")
# ############################################################################ #
# ###################### Simulated sources from T-RECs ####################### #
# ############################################################################ #
trecs = None
if cfg.sky_model.extragalactic.artifical_component.include:
    LOGGER.info("Incorporating T-RECS artificial sources into low-SNR "
                "extragalactic foreground component")
    fits_trecs = None
    if cfg.sky_model.extragalactic.artifical_component.create:
        errh.raise_error(NotImplementedError,
                         "Currently can only load model from fits")
    else:
        if cfg.sky_model.extragalactic.artifical_component.image == "":
            fits_trecs = farm.data.FILES['IMAGES']['TRECS']
        elif cfg.sky_model.extragalactic.artifical_component.image.exists():
            fits_trecs = cfg.sky_model.extragalactic.artifical_component.image
        else:
            errh.raise_error(FileNotFoundError,
                             f"{str(cfg.sky_model.extragalactic.artifical_component.image)} "
                             "does not exist")
        # Parse .fits table data
        # TODO: Conditional here as to whether to load from the fits
        #  image or table
        # Create SkyComponent instance and save .fits image of TRECS sources
        # Have to regrid fits_trecs here before instantiating a SkyComponent from it
        # as memory runs out with the full TRECS image
        trecs = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_trecs,
            name='TRECS',
            coord0=cfg.field.coord0,
            freqs=cfg.correlator.frequencies
        )

    if trecs:
        trecs = trecs.regrid(sky_model)
        sky_model.add_component(trecs)

else:
    LOGGER.info("Not including T-RECS sources into foreground")
# ############################################################################ #
# ############################# EoR H-21cm signal ############################ #
# ############################################################################ #
h21cm = None
if cfg.sky_model.h21cm:
    LOGGER.info("Including EoR 21cm signal into sky model")
    if cfg.sky_model.h21cm.create:
        errh.raise_error(NotImplementedError,
                         "Currently can only load model from fits")
        sys.exit()
    else:
        fits_h21cm = cfg.sky_model.h21cm.image
        if not (fits_h21cm.exists() and fits_h21cm.is_file()):
            errh.raise_error(FileNotFoundError,
                             f"{str(cfg.sky_model.h21cm.image)} "
                             "does not exist")
    LOGGER.info(f"Loading EoR 21cm component from {fits_h21cm}")

    h21cm = farm.sky_model.SkyComponent.load_from_fits(
        fitsfile=fits_h21cm,
        name='H21CM',
        coord0=cfg.field.coord0,
        freqs=cfg.correlator.frequencies
    )

    h21cm = h21cm.regrid(sky_model)
    sky_model.add_component(h21cm)
else:
    LOGGER.info("Not including EoR 21cm signal into sky model")
# ############################################################################ #
# ######## Add all components derived from .fits images to SkyModel, ######### #
# ###### write .fits images for image-derived components, and combine ######## #
# ############### side-lobe and fov Oskar.Sky instances ###################### #
# ############################################################################ #
LOGGER.info("Adding all included sky components into sky model")
for component in sky_model.components:
    if not dryrun:
        component.write_fits(
            cfg.output_dcy / f"{component.name}_component.fits",
            unit='JY/PIXEL'
        )
    else:
        LOGGER.info(f"DRYRUN: Skipping .fits creation for SkyModel components")

if not dryrun:
    sky_model.write_fits(cfg.sky_model.image, unit='JY/PIXEL')
else:
    LOGGER.info("DRYRUN: Skipping .fits creation for SkyModel")

if not model_only:
    # Do not use copy.deepcopy on Sky instances. Throws the error:
    #     TypeError: unsupported operand type(s) for +: 'Sky' and 'int'
    sky = sky_fov.create_copy()
    sky.append(sky_side_lobes)
    LOGGER.info(f"Saving oskar sky model to {cfg.oskar_sky_model_file}")
    sky.save(str(cfg.oskar_sky_model_file))

    cfg.set_oskar_sim_interferometer('sky/oskar_sky_model/file',
                                     cfg.oskar_sky_model_file)

# ############################################################################ #
# ############################ A-Team sources ################################ #
# ############################################################################ #
if cfg.sky_model.ateam and not model_only:
    LOGGER.info("Incorporating A-Team model into field of view")
    ateam_data = farm.data.ATEAM_DATA
    sky_fov.append_sources(**ateam_data)

    # Imperfectly-demixed, residual A-Team sources in side-lobes
    if cfg.sky_model.ateam.demix_error:
        residual_fac = np.ones(len(ateam_data.columns))
        residual_fac.put(ateam_data.columns.get_loc('I'),
                         cfg.sky_model.ateam.demix_error)
        sky_side_lobes.append_sources(**ateam_data * residual_fac)
        LOGGER.info("Incorporating imperfectly demixed A-Team model into "
                    "side lobes")
else:
    LOGGER.info("Not including A-Team sources")
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Synthetic observing run ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
if not model_only:
    # ######################################################################## #
    # ########## Determine scan times and produce diagnostic plots ########### #
    # ######################################################################## #
    scan_times = ()
    if not model_only:
        scan_times = cfg.observation.scan_times(cfg.field.coord0,
                                                cfg.telescope.location,
                                                False)

        msg = 'Computed scan times are:\n'
        for idx, (start, end) in enumerate(scan_times):
            msg += (
                f"\tScan {idx + 1}: {start.strftime('%d%b%Y %H:%M:%S').upper()}"
                f" to {end.strftime('%d%b%Y %H:%M:%S').upper()} "
                f"({(end - start).to_value('s'):.1f}s)\n")
        for line in msg.split('\n'):
            LOGGER.info(line)

        plotting.target_altaz(
            cfg.observation.t_start, cfg.telescope.location,
            cfg.field.coord0, scan_times=scan_times,
            savefig=cfg.output_dcy / "elevation_curve.pdf"
        )
    # ######################################################################## #
    # #################### TEC/Ionospheric calibration/effect ################ #
    # ######################################################################## #
    tecscreen = None
    if cfg.calibration.tec and not model_only:
        if cfg.calibration.tec.create:
            if not dryrun:
                r0 = 1e4
                speed = 150e3 / 3600
                alpha_mag = 0.999
                sampling = 100.0
                t_int = 10.
                layer_params = np.array([(r0, speed, 60.0, 300e3),
                                         (r0, speed / 2.0, -30.0, 310e3)])
                bmax = max(cfg.telescope.baseline_lengths.values())

                screen_width_m = (2 * (np.max(layer_params[:, -1]) *
                                       np.tan(np.radians(
                                           (max(cfg.field.fov) / 2.)))
                                       + bmax))

                # Round screen width to nearest sensible number
                screen_width_m = (
                                             screen_width_m / sampling // 100 + 1) * 100 * sampling

                m = int(bmax / sampling)  # Pixels per sub-aperture
                n = int(screen_width_m / bmax)  # Sub-apertures across screen
                pscale = screen_width_m / (n * m)  # Pixel scale
                rate = t_int ** -1.

                arscreen = farm.calibration.tec.ArScreens(
                    n, m, pscale, rate, layer_params, alpha_mag,
                    cfg.calibration.noise.seed
                )

                tecscreen = farm.calibration.tec.TECScreen(
                    arscreen, cfg.observation.t_start,
                    max([_ for __ in scan_times for _ in __])
                )

                tecscreen.create_tec_screen(cfg.output_dcy / 'tec_screen.fits')
                cfg.calibration.tec.image.append(tecscreen)
            else:
                LOGGER.info("DRYRUN: Skipping TEC screen creation")
        else:
            tec_compatible = check_tec_image_compatibility(
                cfg, cfg.calibration.tec.image
            )
            if not tec_compatible[0]:
                errh.raise_error(ValueError, tec_compatible[1])

            LOGGER.info("Existing TEC .fits images are compatible with scans")
            tecscreen = farm.calibration.tec.TECScreen.create_from_fits(
                cfg.calibration.tec.image, cfg.observation.t_start
            )

    else:
        LOGGER.info("Not including TEC")

    if dryrun or model_only:
        LOGGER.info("Finished, no synthetic observations performed")
        sys.exit()
    # ######################################################################## #
    # ######################### Observation loop ############################# #
    # ######################################################################## #
    LOGGER.info("Running synthetic observations")
    sky_model_mir_im = cfg.sky_model.image.with_suffix('.im')
    sky_model.write_miriad_image(sky_model_mir_im, unit='JY/PIXEL')

    scans = [Scan(start, end) for start, end in scan_times]
    observation = Observation(cfg)
    observation.add_scan(scans)

    measurement_sets = {}
    for scan in observation.scans:
        n_scan = observation.n_scan(scan)
        rseed_scan = observation.generate_scan_seed(cfg.calibration.noise.seed,
                                                    scan)
        n_scan_str = format(n_scan, f'0{len(str(observation.n_scans)) + 1}')

        # Primary beam creation
        scan_beam_fits = cfg.root_name.append(f'_ICUT_{n_scan_str}.fits')
        observation.create_beam_pattern(scan, scan_beam_fits, resample=16,
                                        template=cfg.sky_model.image)

        if tecscreen:
            scan_tec_fits = cfg.root_name.append(f'_TEC_{n_scan_str}.fits')
            observation.get_scan_tec_screen_slice(tecscreen, scan,
                                                  scan_tec_fits)

        # For the image based model cube, it is first necessary to regrid in a
        # way that will allow a single (u,v) grid to represent the data within
        # miriad uvmodel. This is done by taking the lowest frequency as a
        # reference and then scaling the cell size up for higher frequencies.
        # Since the model needs to be in Jy/pixel, it is necessary to reduce
        # the brightness as 1/f**2 to take account of the larger pixel size at
        # higher frequencies.
        sky_model_pbcor = cfg.sky_model.image.stem
        sky_model_pbcor += f"_pbcor_scan{n_scan_str}.fits"
        sky_model_pbcor = cfg.output_dcy / sky_model_pbcor
        pb_multiply(in_image=sky_model_mir_im, pb=scan_beam_fits,
                    out_fitsfile=sky_model_pbcor, cellscal='1/F')
        sky_model_pbcor_mirim = sky_model_pbcor.with_suffix('.mirim')

        scan_out_ms: pathlib.Path = cfg.root_name.append(f'_ICUT_{n_scan_str}.ms')
        scan_out_uvfits = scan_out_ms.with_suffix('.uvfits')
        observation.execute_scan(scan, scan_out_ms)

        # Add SKA1-LOW to measurement set header and export measurement set to
        # uvfits format via casa
        casa.tasks.vishead(vis=f"{scan_out_ms}", mode="put", hdkey="telescope",
                           hdvalue="SKA1-LOW")
        casa.tasks.exportuvfits(vis=f"{scan_out_ms}",
                                fitsfile=f"{scan_out_uvfits}",
                                datacolumn="data", multisource=False,
                                writestation=False, overwrite=True)
        shutil.rmtree(scan_out_ms)

        # Convert to miriad visibility data format and add relevant header
        # information
        temp_scan_out_mirvis = cfg.output_dcy / f'temp_{grc(10)}.mirvis'
        scan_out_mirvis = scan_out_ms.with_suffix('.mirvis')
        miriad.fits(op='uvin', _in=scan_out_uvfits, options="nofq",
                    out=temp_scan_out_mirvis)
        scan_out_uvfits.unlink()

        # Add sky model to visibility data
        miriad.fits(op='xyin', _in=sky_model_pbcor, out=sky_model_pbcor_mirim)
        miriad.uvmodel(vis=temp_scan_out_mirvis, model=sky_model_pbcor_mirim,
                       options="add,zero", out=scan_out_mirvis)
        shutil.rmtree(temp_scan_out_mirvis)

        miriad.puthd(_in=f"{scan_out_mirvis}/restfreq", value=1.42040575)
        miriad.puthd(_in=f"{scan_out_mirvis}/telescop", value="SKA1-LOW")

        # Implement gain and bandpass errors
        # TODO: t_interval hard-coded here
        implement_gain_errors(scan_out_mirvis, t_interval=240.,
                              pnoise=cfg.calibration.gains.phase_err,
                              gnoise=cfg.calibration.gains.amp_err,
                              rseed=observation.products[scan]['seed'])

        implement_bandpass_errors(scan_out_mirvis, nchan=cfg.correlator.n_chan,
                                  freq0=cfg.correlator.freq_min,
                                  chan_width=cfg.correlator.chan_width,
                                  pnoise=cfg.calibration.gains.phase_err,
                                  gnoise=cfg.calibration.gains.amp_err,
                                  rseed=observation.products[scan]['seed'])

        miriad.fits(op='uvout', _in=scan_out_mirvis, out=scan_out_uvfits)
        shutil.rmtree(scan_out_mirvis)

        casa.tasks.importuvfits(vis=f"{scan_out_ms}",
                                fitsfile=f"{scan_out_uvfits}")
        scan_out_uvfits.unlink()

        observation.products[scan]['MS'] = scan_out_ms

    out_ms = cfg.root_name.append('.ms')
    observation.concat_scan_measurement_sets(out_ms)

    # for icut, (t_start, t_end) in enumerate(scan_times):
    #     rseed_icut = int(cfg.calibration.noise.seed +
    #                      icut * (3 - 5 ** 0.5) * 180.)
    #     duration = (t_end - t_start).to_value('s')
    #
    #     # Simulate the primary beam for this scan
    #     sicut = format(icut, f'0{len(str(cfg.observation.n_scan)) + 1}')
    #     scan_beam_fits = cfg.root_name.append(f'_ICUT_{sicut}.fits')
    #     create_beam_pattern_fits(cfg, icut, t_start, duration, scan_beam_fits)
    #     scan_beam_mirim = scan_beam_fits.with_suffix('.im')
    #
    #     # TODO: This is where we left off. Correct below to be farm-compatible
    #     #  code
    #     gsm = pathlib.Path(str(cfg.sky_model.image).rstrip('.fits'))
    #     gsm_cut = gsm.append('_ICUT_' + sicut)
    #     gsm_pcut = gsm.append('_pICUT_' + sicut)
    #     gsm_tcut = gsm.append('_tICUT_' + sicut)
    #
    #     ionof_cut = cfg.calibration.tec.image[icut]
    #     ionot_cut = cfg.root_name.append(
    #         '_iIcut_' + sicut)  # TEC screen for this time cut (miriad image)
    #     ionom_cut = cfg.root_name.append(
    #         '_imIcut_' + sicut)  # TEC screen + residual errors for this time cut (miriad image)
    #     iono_cut = cfg.root_name.append(
    #         '_iIcut_' + sicut + '.fits')  # TEC screen + residual errors for this time cut (.fits image)
    #
    #     cmpt_mscut = cfg.root_name.append('_ICUT_' + sicut + '.ms')
    #     cmpt_msmcut = cfg.root_name.append('_ICUT_' + sicut + '.msm')
    #     cmpt_uvfcut = cfg.root_name.append('_ICUT_' + sicut + '.uvf')
    #     cmpt_uvcut = cfg.root_name.append('_ICUT_' + sicut + '.uv')
    #
    #     out_uvcut = cfg.root_name.append('_ICUT_' + sicut + 'out.uv')
    #     out_mscut = cfg.root_name.append('_ICUT_' + sicut + 'out.ms')
    #     out_uvfcut = cfg.root_name.append('_ICUT_' + sicut + 'out.uvf')
    #
    #     LOGGER.info(f"Multiplying sky model by beam response, {scan_beam_fits}")
    #     miriad.fits(op="xyin", _in=scan_beam_fits, out=scan_beam_mirim)
    #
    #     # For the image based model cube, it is first necessary to regrid in a
    #     # way that will allow a single (u,v) grid to represent the data within
    #     # miriad uvmodel. This is done by taking the lowest frequency as a
    #     # reference and then scaling the cell size up for higher frequencies.
    #     # Since the model needs to be in Jy/pixel, it is necessary to reduce
    #     # the brightness as 1/f**2 to take account of the larger pixel size at
    #     # higher frequencies.
    #     expr = f"<{scan_beam_mirim}>*<{sky_model_mir_im}>/z**2"
    #     zrange = f"1,{cfg.correlator.freq_max / cfg.correlator.freq_min}"
    #     miriad.maths(exp=expr, out=gsm_pcut, zrange=zrange)
    #
    #     text = f'/bin/cp -r {gsm_pcut} {gsm_tcut}'
    #     subprocess.run(text, shell=True)
    #
    #     miriad.puthd(_in=f"{gsm_tcut}/cellscal", value="1/F")
    #     miriad.regrid(_in=gsm_pcut, tin=gsm_tcut, out=gsm_cut)
    #
    #     # Get the relevant cut from the ionospheric model and scale for net
    #     # residual effect
    #     miriad.fits(op="xyin", _in=ionof_cut, out=ionot_cut)
    #     miriad.maths(exp=f"<{ionot_cut}>*{cfg.calibration.tec.err:.3e}",
    #                  out=ionom_cut)
    #     miriad.fits(op="xyout", _in=ionom_cut, out=iono_cut)
    #
    #     # Adjust oskar's sim_interferometer settings in .ini files
    #     cfg.set_oskar_sim_interferometer(
    #         'sky/oskar_sky_model/file', cfg.oskar_sky_model_file
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'observation/start_time_utc',
    #         t_start.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-2]
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'telescope/external_tec_screen/input_fits_file', iono_cut
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'interferometer/ms_filename', cmpt_mscut
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'observation/length', format(duration, '.1f')
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'observation/num_time_steps', int(duration // cfg.correlator.t_int)
    #     )
    #
    #     cfg.set_oskar_sim_interferometer(
    #         'interferometer/noise/seed', format(rseed_icut, '.0f')
    #     )
    #
    #     # TODO: Multiple header insertions for telescope -> SKA1-LOW in newest
    #     # skye_cuts.py need implementing
    #
    #     # Run oskar's sim_interferometer task and produce measurement set
    #     run_oskar_sim_interferometer(cfg.sinterferometer_ini)
    #
    #     # Add SKA1-LOW to measurement set header and export measurement set to
    #     # uvfits format via casa
    #     casa.tasks.vishead(vis=f"{cmpt_mscut}", mode="put", hdkey="telescope",
    #                        hdvalue="SKA1-LOW")
    #     casa.tasks.exportuvfits(vis=f"{cmpt_mscut}", fitsfile=f"{cmpt_uvfcut}",
    #                             datacolumn="data", multisource=False,
    #                             writestation=False, overwrite=True)
    #
    #     miriad.fits(op='uvin', _in=cmpt_uvfcut, options="nofq", out=cmpt_uvcut)
    #     miriad.uvmodel(vis=cmpt_uvcut, model=gsm_cut, options="add,zero",
    #                    out=out_uvcut)
    #
    #     # TODO: t_interval hard-coded
    #     implement_gain_errors(out_uvcut, t_interval=240.,
    #                           pnoise=cfg.calibration.gains.phase_err,
    #                           gnoise=cfg.calibration.gains.amp_err,
    #                           rseed=rseed_icut)
    #
    #     implement_bandpass_errors(out_uvcut, nchan=cfg.correlator.n_chan,
    #                               pnoise=cfg.calibration.gains.phase_err,
    #                               gnoise=cfg.calibration.gains.amp_err,
    #                               rseed=rseed_icut)
    #
    #     miriad.fits(op='uvout', _in=out_uvcut, out=out_uvfcut)
    #
    #     casa.tasks.importuvfits(vis=f"{out_mscut}", fitsfile=f"{out_uvfcut}")
    #     measurement_sets.append(out_mscut)

    # Concatenate all measurement sets produced in the above for-loop using casa
    # so that a single final visibility dataset is produced for the challenge
    # casa.tasks.concat(vis=[str(ms) for ms in measurement_sets],
    #                   concatvis=f"{cfg.root_name}.ms",
    #                   timesort=True)

    gaussian_taper_arcsec = 60.  # in arcsec
    niter = 10000

    wsclean_args = {
        'weight': 'uniform',
        'taper-gaussian': f'{gaussian_taper_arcsec}asec',
        'super-weight': 4,
        'name': f'{cfg.root_name}_wsclean',
        'size': f'{cfg.field.nx} {cfg.field.ny}',
        'scale': f'{cfg.field.cdelt * 3600}asec',
        'channels-out': cfg.correlator.n_chan,
        'niter': niter,
        'pol': 'xx'
    }
    farm.software.wsclean(f"{cfg.root_name}.ms", wsclean_args,
                          consolidate_channels=True)

# Ionospheric simulation altered to 4 hour total, sliced into relevant parts to
# correspond with scans

# TODO: Explore oskar options for parallelisation and GPU use?
# TODO: Sanity checks need implementing at each stage
# TODO: All-sky view of strong sources
# TODO: Remove any intermediate data-products

# ############################################################################ #
# p_k_field, bins_field = pbox.get_power(imdata_gssm[0], (0.002, 0.002,))
#
# plt.close('all')
#
# plt.plot(bins_field, p_k_field, 'b-')
#
# plt.xscale('log')
# plt.yscale('log')
#
# plt.show()

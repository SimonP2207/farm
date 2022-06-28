#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import logging
import argparse
import pathlib
from typing import Union, Tuple, List, Optional
from datetime import datetime

from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.time import Time

import farm
import farm.data.loader as loader
import farm.physics.astronomy as ast
import farm.miscellaneous as misc
import farm.miscellaneous.error_handling as errh
import farm.miscellaneous.plotting as plotting
import farm.sky_model.tb_functions as tb_funcs
from farm.software import casa
from farm.software.miriad import miriad
from farm.software import oskar
from farm.software.oskar import (run_oskar_sim_beam_pattern,
                                 run_oskar_sim_interferometer)
from farm import LOGGER


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


def determine_scan_times(farm_cfg: loader.FarmConfiguration,
                         save_plot: Union[bool, pathlib.Path],
                         logger: Optional[logging.Logger] = None
                         ) -> Tuple[Tuple[Time, Time], ...]:
    """
    Calculates and plots scan times for a farm configuration. Also optionally
    logs results/operations

    Parameters
    ----------
    farm_cfg
        FarmConfiguration instance to parse information from
    save_plot
        Whether to plot the scan times as an altitude/azimuth plot over the
        duration of the obseration. If False, no plot is produced, otherwise a
        pathlib.Path must be given to save the resulting plot to
    logger
        logging.Logger instance to log messages to

    Returns
    -------
    Tuple containing two-tuples of (astropy.time.Time, astropy.time.Time)
    representing scan start/end times
    """
    if logger:
        logger.info("Computing scan times")

    # Compute scan times
    scans = ast.scan_times(
        farm_cfg.observation.time, farm_cfg.field.coord0,
        farm_cfg.telescope.location, farm_cfg.observation.n_scan,
        farm_cfg.observation.t_total, farm_cfg.observation.min_elevation,
        farm_cfg.observation.min_gap_scan, partial_scans_allowed=False
    )

    if logger:
        msg = 'Computed scan times are:\n'
        for idx, (start, end) in enumerate(scans):
            msg += (
                f"\tScan {idx + 1}: {start.strftime('%d%b%Y %H:%M:%S').upper()}"
                f" to {end.strftime('%d%b%Y %H:%M:%S').upper()} "
                f"({(end - start).to_value('s'):.1f}s)\n")
        for line in msg.split('\n'):
            logger.info(line)

    # Produce plot of elevation curve and proposed scans
    if save_plot:
        logger.info(f"Saving elevation plot to {save_plot}")
        plotting.target_altaz(
            farm_cfg.observation.time, farm_cfg.telescope.location,
            farm_cfg.field.coord0, scan_times=scans,
            savefig=save_plot
        )

    return tuple(scans)


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
    model_only = True
    log_level = logging.DEBUG

cfg = loader.FarmConfiguration(config_file)
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
# ############ Determine scan times and produce diagnostic plots ############# #
# ############################################################################ #
scan_times = ()
if not model_only:
    scan_times = determine_scan_times(
        cfg, cfg.output_dcy / "elevation_curve.pdf", LOGGER
    )
# ############################################################################ #
# ###################### TEC/Ionospheric calibration/effect ################## #
# ############################################################################ #
if cfg.calibration.tec and not model_only:
    if cfg.calibration.tec.create:
        cfg.calibration.tec.image.append(
            farm.calibration.tec.create_tec_screens(
                cfg, scan_times, tec_prefix='iono_tec', logger=LOGGER)
        )
    else:
        tec_compatible = check_tec_image_compatibility(
            cfg, cfg.calibration.tec.image
        )
        if not tec_compatible[0]:
            errh.raise_error(ValueError, tec_compatible[1])

        LOGGER.info("Existing of TEC .fits images are compatible with scans")
else:
    LOGGER.info("Not including TEC")
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
            fitsfile=fits_gdsm, name='GDSM', freqs=cfg.correlator.frequencies
        )
        gdsm = gdsm.regrid(sky_model)
    # In the case that the small-scale Galactic emission is not included in
    # the SkyModel, add the large-scale emission component alone to the
    # SkyModel
    if not cfg.sky_model.galactic.small_scale_component:
        sky_model.add_component(gdsm)
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
        merged = gssm.merge(gdsm, (gssm.cdelt, gssm.cdelt, 0.),
                            (56. / 60., 56. / 60., 0.), 'GASM')
        sky_model.add_component(merged)
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
        if misc.fits.is_fits_table(fits_eg_real):
            # Parse .fits table data
            data = misc.fits.fits_table_to_dataframe(fits_eg_real)
        elif misc.file_handling.is_osm_table(fits_eg_real):
            data = misc.file_handling.osm_to_dataframe(fits_eg_real)
        elif misc.fits.is_fits_image(fits_eg_real):
            errh.raise_error(NotImplementedError,
                             "Can't load known points sources from .fits image")
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
            beam={'maj': 2. * 60, 'min': 2. * 60., 'pa': 0.}
        )
        eg_real.write_fits(
            cfg.output_dcy / f"{eg_real.name}_component.fits",
            unit='JY/PIXEL'
        )

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
    component.write_fits(
        cfg.output_dcy / f"{component.name}_component.fits",
        unit='JY/PIXEL'
    )
sky_model.write_fits(cfg.sky_model.image, unit='JY/PIXEL')

if not model_only:
    sky_model_mir_im = cfg.sky_model.image.with_suffix('.im')
    sky_model.write_miriad_image(sky_model_mir_im, unit='JY/PIXEL')
    # Do not use copy.deepcopy on Sky instances. Throws the error:
    #     TypeError: unsupported operand type(s) for +: 'Sky' and 'int'
    sky = sky_fov.create_copy()
    sky.append(sky_side_lobes)
    LOGGER.info(f"Saving oskar sky model to {cfg.oskar_sky_model_file}")
    sky.save(str(cfg.oskar_sky_model_file))

    cfg.set_oskar_sim_interferometer('sky/oskar_sky_model/file',
                                     cfg.oskar_sky_model_file)
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Synthetic observing run ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
    LOGGER.info("Running synthetic observations")

    from dataclasses import dataclass, field
    from collections.abc import Iterable

    @dataclass(frozen=True)
    class Scan:
        """Class for an observational scan"""
        start_time: Time
        end_time: Time
        duration: float = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, 'duration',
                               (self.end_time - self.start_time).to_value('s'))


    class Observation:
        """Class for an entire observational run incorporating multiple scans"""
        def __init__(self):
            self._scans = []

        @property
        def scans(self) -> List[Scan]:
            """List of scans in observation"""
            return self._scans

        def add_scan(self, new_scans):
            """Add a scan"""
            if isinstance(new_scans, Iterable):
                for scan in new_scans:
                    self.add_scan(scan)
            else:
                if not isinstance(new_scans, Scan):
                    errh.raise_error(TypeError,
                                     "Can only add Scan instances to "
                                     "Observation instance, not "
                                     f"{type(new_scans)} instance")
                self._scans.append(new_scans)

        @property
        def total_time(self) -> float:
            """Total time on source, over the course of the observation [s]"""
            return sum([scan.duration for scan in self.scans])


    measurement_sets = []
    for icut, (t_start, t_end) in enumerate(scan_times):
        duration = (t_end - t_start).to_value('s')

        # Simulate the primary beam for this scan
        sicut = format(icut, f'0{len(str(cfg.observation.n_scan)) + 1}')
        scan_beam_fits = cfg.root_name.append('_ICUT_{sicut}.fits')
        create_beam_pattern_fits(cfg, icut, t_start, duration, scan_beam_fits)
        scan_beam_mirim = scan_beam_fits.with_suffix('.im')

        # TODO: This is where we left off. Correct below to be farm-compatible
        #  code
        gsm = pathlib.Path(str(cfg.sky_model.image).rstrip('.fits'))
        gsm_cut = gsm.append('_ICUT_' + sicut)
        gsm_pcut = gsm.append('_pICUT_' + sicut)
        gsm_tcut = gsm.append('_tICUT_' + sicut)

        ionof_cut = cfg.calibration.tec.image[icut]
        ionot_cut = cfg.root_name.append('_iIcut_' + sicut)  # TEC screen for this time cut (miriad image)
        ionom_cut = cfg.root_name.append('_imIcut_' + sicut)  # TEC screen + residual errors for this time cut (miriad image)
        iono_cut = cfg.root_name.append('_iIcut_' + sicut + '.fits')  # TEC screen + residual errors for this time cut (.fits image)

        cmpt_mscut = cfg.root_name.append('_ICUT_' + sicut + '.ms')
        cmpt_msmcut = cfg.root_name.append('_ICUT_' + sicut + '.msm')
        cmpt_uvfcut = cfg.root_name.append('_ICUT_' + sicut + '.uvf')
        cmpt_uvcut = cfg.root_name.append('_ICUT_' + sicut + '.uv')

        out_uvcut = cfg.root_name.append('_ICUT_' + sicut + 'out.uv')
        out_mscut = cfg.root_name.append('_ICUT_' + sicut + 'out.ms')
        out_uvfcut = cfg.root_name.append('_ICUT_' + sicut + 'out.uvf')

        LOGGER.info(f"Multiplying sky model by beam response, {scan_beam_fits}")
        miriad.fits(op="xyin", _in=scan_beam_fits, out=scan_beam_mirim)
        expr = f"<{scan_beam_mirim}>*<{sky_model_mir_im}>"
        miriad.maths(exp=expr, out=gsm_pcut)

        text = f'/bin/cp -r {gsm_pcut} {gsm_tcut}'
        subprocess.run(text, shell=True)

        # For the image based model cube, it is first necessary to regrid in a
        # way that will allow a single (u,v) grid to represent the data within
        # miriad uvmodel. This is done by taking the lowest frequency as a
        # reference and then scaling the cell size up for higher frequencies.
        # Since the model needs to be in Jy/pixel, it is necessary to reduce the
        # brightness as 1/f**2 to take account of the larger pixel size at
        # higher frequencies.
        miriad.puthd(_in=f"{gsm_tcut}/cellscal", value="1/F")
        miriad.regrid(_in=gsm_pcut, tin=gsm_tcut, out=gsm_cut)

        # Get the relevant cut from the ionospheric model and scale for net
        # residual effect
        miriad.fits(op="xyin", _in=ionof_cut, out=ionot_cut)
        miriad.maths(exp=f"<{ionot_cut}>*{cfg.calibration.tec.err:.3e}",
                     out=ionom_cut)
        miriad.fits(op="xyout", _in=ionom_cut, out=iono_cut)

        # Adjust oskar's sim_interferometer settings in .ini file
        cfg.set_oskar_sim_interferometer(
            'sky/oskar_sky_model/file', cfg.oskar_sky_model_file
        )

        cfg.set_oskar_sim_interferometer(
            'observation/start_time_utc',
            t_start.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-2]
        )

        cfg.set_oskar_sim_interferometer(
            'telescope/external_tec_screen/input_fits_file', iono_cut
         )

        cfg.set_oskar_sim_interferometer(
            'interferometer/ms_filename', cmpt_mscut
        )

        cfg.set_oskar_sim_interferometer(
            'observation/length', format(duration, '.1f')
        )

        cfg.set_oskar_sim_interferometer(
            'observation/num_time_steps', int(duration // cfg.correlator.t_int)
        )

        # Run oskar's sim_interferometer task and produce measurement set
        run_oskar_sim_interferometer(cfg.sinterferometer_ini)

        # Add SKA1-LOW to measurement set header and export measurement set to
        # uvfits format via casa
        casa.tasks.vishead(vis=f"{cmpt_mscut}", mode="put", hdkey="telescope",
                           hdvalue="SKA1-LOW")
        casa.tasks.exportuvfits(vis=f"{cmpt_mscut}", fitsfile=f"{cmpt_uvfcut}",
                                datacolumn="data", multisource=False,
                                writestation=False, overwrite=True)

        miriad.fits(op='uvin', _in=cmpt_uvfcut, options="nofq", out=cmpt_uvcut)
        miriad.uvmodel(vis=cmpt_uvcut, model=gsm_cut, options="add,zero",
                       out=out_uvcut)
        # TODO: Move gperror to python functionality? Initialise gain tables
        #  with gperror, then manipulate that table with python
        miriad.gperror(vis=out_uvcut, interval=1.,
                       pnoise=cfg.calibration.gains.phase_err,
                       gnoise=cfg.calibration.gains.amp_err)

        miriad.fits(op='uvout', _in=out_uvcut, out=out_uvfcut)

        casa.tasks.importuvfits(vis=f"{out_mscut}", fitsfile=f"{out_uvfcut}")
        measurement_sets.append(out_mscut)

    # Concatenate all measurement sets produced in the above for-loop using casa
    # so that a single final visibility dataset is produced for the challenge
    casa.tasks.concat(vis=str(measurement_sets),
                      concatvis=f"{cfg.root_name}.ms",
                      timesort=True)

    gaussian_taper_arcsec = 60.  # in arcsec
    niter = 10000

    wsclean_args ={
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

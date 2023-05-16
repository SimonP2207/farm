#!/usr/bin/env python3
"""
Pipeline for the creation of a synthetic dataset, and its corresponding
deconvolved image, containing the epoch of reionisation's hydrogen 21cm signal
"""
import os
import shutil
import sys
import logging
import argparse
import pathlib
from datetime import datetime
from typing import TypeVar, Optional

import h5py
import numpy as np
from astropy.io import fits

import farm
from farm import config
from farm.calibration.noise import generate_gain_errors
import farm.physics.astronomy as ast
import farm.miscellaneous as misc
from farm.miscellaneous import generate_random_chars as grc
import farm.miscellaneous.error_handling as errh
from farm.miscellaneous.image_functions import pb_multiply
from farm.miscellaneous import plotting
from farm.observing import Scan, Observation, Field, Correlator, Telescope
import farm.sky_model.tb_functions as tb_funcs
from farm.software import casa
from farm.software.miriad import miriad
from farm.software import oskar
from farm import LOGGER

AnySkyCompCfg = TypeVar('AnySkyCompCfg', bound=config.SkyComponentConfiguration)


# TODO: Generic function to process a single SkyComponent from the
#  configuration. Still in progress.
def process_component(
        comp_name: str,
        sky_model: farm.sky_model.SubbandSkyModel,
        comp_cfg: AnySkyCompCfg,
        field: config.Field,
        correlator: config.Correlator
) -> Optional[farm.sky_model.SkyComponent]:
    """
    Method to process a SkyComponent according to its configuration

    Parameters
    ----------
    comp_name
        Name to give the SkyComponent instance
    sky_model
        SubbandSkyModel instance with which to regrid to
    comp_cfg
        farm.config.SkyComponentConfiguration or any of its childclasses from
        which to process the SkyComponent
    field
        farm.config.Field configuration instance
    correlator
        farm.config.Correlator configuration instance

    Returns
    -------
    If SkyComponent configuration is included, returns a
    farm.sky_model.SkyComponent instance, otherwise None
    """
    # Include the sky component in the sky model?
    comp = None
    if comp_cfg.include:
        # Whether to create from scratch or load from table/image
        if not hasattr(comp_cfg, 'create') or not comp_cfg.create:
            # Load SkyComponent from image or table
            if not comp_cfg.image.exists():
                errh.raise_error(FileNotFoundError,
                                 f"{comp_cfg.image} does not exist")
            elif comp_cfg.image.is_dir():
                errh.raise_error(FileNotFoundError,
                                 f"{comp_cfg.image} is a directory")

            LOGGER.info(
                f"Loading {comp_name} component from {comp_cfg.image}")

            if misc.fits.is_fits_image(comp_cfg.image):
                with fits.open(comp_cfg.image) as hdulist:
                    fits_nx = np.shape(hdulist[0].data)[-1]

                # cdelt/coord0 replace fits header values here since small-scale
                # image is notx  a real observation -< Is this a problem?
                comp = farm.sky_model.SkyComponent.load_from_fits(
                    fitsfile=comp_cfg.image,
                    name=comp_name,
                    cdelt=field.fov[0] / fits_nx,
                    coord0=field.coord0,
                    freqs=correlator.frequencies
                )
            elif misc.fits.is_fits_table(comp_cfg.image):
                if misc.fits.is_fits_table(fits_eg_real):
                    # Parse .fits table data
                    data = misc.fits.fits_table_to_dataframe(fits_eg_real)
                elif misc.file_handling.is_osm_table(fits_eg_real):
                    data = misc.file_handling.osm_to_dataframe(fits_eg_real)
                else:
                    errh.raise_error(Exception,
                                     f"Unsure of format for table, "
                                     f"{fits_eg_real}")
                # TODO: How to implement source filtering here?
                ...
            else:
                errh.raise_error(ValueError,
                                 f"{comp_cfg.image} is neither a fits image, "
                                 f"fits table, or OSKAR sky model file")

        else:
            # Create SkyComponent from scratch
            ...

        return comp.regrid(sky_model)


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
    parser.add_argument("-r", "--dry-run",
                        help="Complete a dry run to check for validity",
                        action="store_true")
    args = parser.parse_args()
    config_file = pathlib.Path(args.config_file)
    MODEL_ONLY = args.model_only
    DRY_RUN = args.dry_run
    LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
else:
    config_file = pathlib.Path(farm.data.FILES['EXAMPLE_CONFIG'])
    MODEL_ONLY = False
    DRY_RUN = False
    LOG_LEVEL = logging.DEBUG

cfg = config.load_configuration_from_toml(config_file)
root_name = cfg['directories']['root_name']
output_dcy = pathlib.Path(cfg['directories']['output_dcy'])

if len(sys.argv) != 1:
    os.chdir(output_dcy)

# ############################################################################ #
# ######################## Set up the logger ################################# #
# ############################################################################ #
pipeline_start = datetime.now()
logfile_name = f'farm{pipeline_start.strftime("%Y%b%d_%H%M%S").upper()}.log'
logfile = output_dcy / logfile_name
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
console_handler.setLevel(LOG_LEVEL)
LOGGER.addHandler(console_handler)
# ############################################################################ #
# ########### Create run's root directory if it doesn't exists ############### #
# ############################################################################ #
if not output_dcy.exists():
    LOGGER.info("Creating output directory %s" % output_dcy)
    output_dcy.mkdir(exist_ok=True)
# ############################################################################ #
field = Field(**cfg['observation']['field'])
tscop = Telescope(pathlib.Path(cfg["directories"]['telescope_model']))
correlator = Correlator(**cfg['observation']['correlator'])
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Initial set up of OSKAR ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
sky_fov, sky_side_lobes = None, None
if not MODEL_ONLY:
    LOGGER.info("Setting up Oskar Sky instances")
    sky_fov = oskar.Sky()
    sky_side_lobes = oskar.Sky()
# ############################################################################ #
# ############################################################################ #
# #####################  # # # # # # # # # # # # # # # ####################### #
# ########################### SkyModel Creation ############################## #
# #####################  # # # # # # # # # # # # # # # ####################### #
# ############################################################################ #
LOGGER.info(f"Instantiating SubbandSkyModel")
sky_model = farm.sky_model.SubbandSkyModel((field.nx, field.ny),
                                           field.cdelt, field.coord0,
                                           correlator.frequencies)
# ############################################################################ #
# ################## Large-scale Galactic foreground model ################### #
# ############################################################################ #
GDSM = None
if cfg.sky_model.galactic.large_scale_component.include:
    if cfg.sky_model.galactic.large_scale_component.create:
        LOGGER.info("Creating large-scale Galactic foreground images")
        GDSM = farm.sky_model.SkyComponent(
            name='GDSM', npix=(field.nx, field.ny),
            cdelt=field.cdelt, coord0=field.coord0,
            tb_func=tb_funcs.gdsm2016_t_b
        )
        GDSM.add_frequency(correlator.frequencies)
    else:
        fits_gdsm = cfg.sky_model.galactic.small_scale_component.image
        if not fits_gdsm.exists():
            errh.raise_error(FileNotFoundError, f"{fits_gdsm} does not exist")
        elif fits_gdsm.is_dir():
            errh.raise_error(FileNotFoundError, f"{fits_gdsm} is a directory")

        LOGGER.info(f"Loading Galactic large-scale component from {fits_gdsm}")
        GDSM = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_gdsm, name='GDSM',
            freqs=correlator.frequencies
        )
        GDSM = GDSM.regrid(sky_model)
    # In the case that the small-scale Galactic emission is not included in
    # the SkyModel, add the large-scale emission component alone to the
    # SkyModel
    if not cfg.sky_model.galactic.small_scale_component:
        sky_model.add_component(GDSM)
    else:
        if DRY_RUN:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {GDSM.name} "
                        f"SkyComponent")
        else:
            GDSM.write_fits(output_dcy / f"{GDSM.name}_component.fits",
                            unit='JY/PIXEL')
else:
    LOGGER.info("Not including large-scale Galactic component")
# ############################################################################ #
# ################ Small-scale Galactic foreground model ##################### #
# ############################################################################ #
GSSM = None
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
        GSSM = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_gssm,
            name='GSSM',
            cdelt=field.fov[0] / fits_nx,
            coord0=field.coord0,
            freqs=correlator.frequencies
        )

        rot_angle = ast.angle_to_galactic_plane(field.coord0)
        LOGGER.info(f"Rotating Galactic small-scale component by "
                    f"{np.degrees(rot_angle):.1f}deg")
        GSSM.rotate(angle=rot_angle, inplace=True)

        LOGGER.info("Regridding small-scale image")
        GSSM = GSSM.regrid(sky_model)

    if cfg.sky_model.galactic.large_scale_component.include:
        LOGGER.info("Normalising small/large-scale power spectra")
        GSSM.normalise(GDSM, inplace=True)
        # TODO: Hard-coded beam information here (56 arcmin)
        merged = GSSM.merge(GDSM, (GSSM.cdelt, GSSM.cdelt, 0.),
                            (56. / 60., 56. / 60., 0.), 'GASM')
        sky_model.add_component(merged)
        if DRY_RUN:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {GSSM.name} "
                        f"SkyComponent")
        else:
            GSSM.write_fits(output_dcy / f"{GSSM.name}_component.fits",
                            unit='JY/PIXEL')
    else:
        sky_model.add_component(GSSM)
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
EG_REAL = None
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
            field.fov, field.coord0.ra.deg,
            field.coord0.dec.deg, data.ra, data.dec
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
        EG_REAL = farm.sky_model.SkyComponent.load_from_dataframe(
            data, 'EG_Known', field.cdelt,
            field.coord0, fov=field.fov,
            freqs=correlator.frequencies,
            flux_range=(cfg_eg_real.flux_transition,
                        cfg_eg_real.flux_inner),
            beam=None
        )

        if not DRY_RUN:
            EG_REAL.write_fits(
                output_dcy / f"{EG_REAL.name}_component.fits",
                unit='JY/PIXEL'
            )
        else:
            LOGGER.info(f"DRYRUN: Skipping .fits creation for {EG_REAL.name} "
                        f"SkyComponent")

        # Below line commented out so that any Gaussian sources are added to
        # Oskar to deal with
        # sky_model.add_component(eg_real)

        if not MODEL_ONLY:
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
TRECS = None
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
            errh.raise_error(
                FileNotFoundError,
                f"{str(cfg.sky_model.extragalactic.artifical_component.image)} "
                "does not exist"
            )
        # Parse .fits table data
        # TODO: Conditional here as to whether to load from the fits
        #  image or table
        # Create SkyComponent instance and save .fits image of TRECS sources
        # Have to regrid fits_trecs here before instantiating a SkyComponent
        # from it as memory runs out with the full TRECS image
        TRECS = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_trecs,
            name='TRECS',
            coord0=field.coord0,
            freqs=correlator.frequencies
        )

    if TRECS:
        TRECS = TRECS.regrid(sky_model)
        sky_model.add_component(TRECS)

else:
    LOGGER.info("Not including T-RECS sources into foreground")
# ############################################################################ #
# ############################# EoR H-21cm signal ############################ #
# ############################################################################ #
H21CM = None
if cfg.sky_model.h21cm:
    LOGGER.info("Including EoR 21cm signal into sky model")
    fits_h21cm = cfg.sky_model.h21cm.image
    if cfg.sky_model.h21cm.create:
        errh.raise_error(NotImplementedError,
                         "Currently can only load model from fits")
    else:
        if not (fits_h21cm.exists() and fits_h21cm.is_file()):
            errh.raise_error(FileNotFoundError,
                             f"{str(cfg.sky_model.h21cm.image)} "
                             "does not exist")
    LOGGER.info(f"Loading EoR 21cm component from {fits_h21cm}")

    H21CM = farm.sky_model.SkyComponent.load_from_fits(
        fitsfile=fits_h21cm,
        name='H21CM',
        coord0=field.coord0,
        freqs=correlator.frequencies
    )

    H21CM = H21CM.regrid(sky_model)
    sky_model.add_component(H21CM)
else:
    LOGGER.info("Not including EoR 21cm signal into sky model")
# ############################################################################ #
# ######## Add all components derived from .fits images to SkyModel, ######### #
# ###### write .fits images for image-derived components, and combine ######## #
# ############### side-lobe and fov Oskar.Sky instances ###################### #
# ############################################################################ #
LOGGER.info("Adding all included sky components into sky model")
for component in sky_model.components:
    if not DRY_RUN:
        component.write_fits(
            output_dcy / f"{component.name}_component.fits",
            unit='JY/PIXEL'
        )
    else:
        LOGGER.info("DRYRUN: Skipping .fits creation for SubbandSkyModel components")

if not DRY_RUN:
    sky_model.write_fits(cfg.sky_model.image, unit='JY/PIXEL')
else:
    LOGGER.info("DRYRUN: Skipping .fits creation for SubbandSkyModel")

if not MODEL_ONLY:
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
if cfg.sky_model.ateam and not MODEL_ONLY:
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
# ############################################################################ #
# ############## Determine scan times and produce diagnostic plots ########### #
# ############################################################################ #
scan_times = ()
if not MODEL_ONLY:
    scan_times = cfg.observation.scan_times(field.coord0,
                                            tscop.location,
                                            False)

    msg = 'Computed scan times are:\n'
    for idx, (start, end) in enumerate(scan_times):
        msg += (
            f"\tScan {idx + 1}: {start.strftime('%d%b%Y %H:%M:%S').upper()}"
            f" to {end.strftime('%d%b%Y %H:%M:%S').upper()} "
            f"({(end - start).to_value('s'):.1f}s)")
        msg += '' if idx == (len(scan_times) - 1) else '\n'
    for line in msg.split('\n'):
        LOGGER.info(line)

    plotting.target_altaz(
        cfg.observation.t_start, tscop.location,
        field.coord0, scan_times=scan_times,
        savefig=output_dcy / "elevation_curve.pdf"
    )
# ############################################################################ #
# ######################## TEC/Ionospheric calibration/effect ################ #
# ############################################################################ #
tecscreen = None
if cfg.calibration.tec and not MODEL_ONLY:
    if cfg.calibration.tec.create:
        if not DRY_RUN:
            alpha_mag = cfg['calibration']['TEC']['alpha_mag']
            sampling = cfg['calibration']['TEC']['pixel_size']
            t_int = cfg['calibration']['TEC']['t_frame']

            layer_params = np.empty((0, 4))
            props = ('r0', 'vel', 'direction', 'altitude')
            for layer in cfg['calibration']['TEC']['layers']:
                layer_params = np.vstack(
                    (layer_params, [layer[prop] for prop in props])
                )

            bmax = max(tscop.baseline_lengths.values())

            tecscreen = farm.calibration.tec.TECScreen.create_from_params(
                t_start=cfg.observation.t_start,
                t_end=max([_ for __ in scan_times for _ in __]),
                t_int=t_int, pixel_m=sampling, fov=max(field.fov),
                bmax=bmax, layer_params=layer_params, alpha_mag=alpha_mag,
                rseed=cfg.calibration.noise.seed
            )

            tecscreen.create_tec_screen(output_dcy / 'tec_screen.fits')
            cfg.calibration.tec.image.append(tecscreen)
        else:
            LOGGER.info("DRYRUN: Skipping TEC screen creation")
    else:
        tec_compatible = farm.config.check_tec_image_compatibility(
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

if DRY_RUN or MODEL_ONLY:
    LOGGER.info("Finished, no synthetic observations performed")
    sys.exit()
# ############################################################################ #
# ############################# Observation loop ############################# #
# ############################################################################ #
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

    # Implement gain/bandpass errors for scan
    # See https://ska-telescope.gitlab.io/sim/oskar/telescope_model/telescope_model.html#telescope-gain-model
    # for description of how produced gain errors are implemented as part of the
    # telescope model
    num_int, num_freq, num_tel = (np.ceil(scan.duration / correlator.t_int),
                                  correlator.n_chan,
                                  tscop.n_stations)

    # Distribution parameters.
    t_beta = cfg.calibration.gains.beta
    t_mean_amp = cfg.calibration.gains.amp_mean
    t_mean_phase = cfg.calibration.gains.phase_mean
    t_std_amp = cfg.calibration.gains.amp_err
    t_std_phase = cfg.calibration.gains.phase_err

    f_beta = cfg.calibration.bandpass.beta
    f_mean_amp = cfg.calibration.bandpass.amp_mean
    f_mean_phase = cfg.calibration.bandpass.phase_mean
    f_std_amp = cfg.calibration.bandpass.amp_err
    f_std_phase = cfg.calibration.bandpass.phase_err

    gains = generate_gain_errors(
        num_int, num_freq, num_tel, rseed_scan,
        t_beta, t_mean_amp, t_mean_phase, t_std_amp, t_std_phase,
        f_beta, f_mean_amp, f_mean_phase, f_std_amp, f_std_phase,
    )

    # Write HDF5 file with recognised dataset names.
    tscop_gains_file = tscop.model / "gain_model.h5"
    if tscop_gains_file.exists():
        LOGGER.info(f"{tscop_gains_file.resolve()} exists. Removing")
        tscop_gains_file.unlink()
    with h5py.File(tscop_gains_file, "w") as hdf_file:
        hdf_file.create_dataset("freq (Hz)", data=correlator.frequencies)
        hdf_file.create_dataset("gain_xpol", data=gains)
    gains_save_dcy = output_dcy / 'gains'
    shutil.copy2(tscop_gains_file, gains_save_dcy / f'gain_scan{n_scan_str}.h5')

    # Primary beam creation
    scan_beam_fits = root_name.append(f'_ICUT_{n_scan_str}.fits')
    observation.create_beam_pattern(scan, scan_beam_fits, resample=16,
                                    template=cfg.sky_model.image)

    if tecscreen:
        scan_tec_fits = root_name.append(f'_TEC_{n_scan_str}.fits')
        observation.get_scan_tec_screen_slice(tecscreen, scan, scan_tec_fits)

    # For the image based model cube, it is first necessary to regrid in a
    # way that will allow a single (u,v) grid to represent the data within
    # miriad uvmodel. This is done by taking the lowest frequency as a
    # reference and then scaling the cell size up for higher frequencies.
    # Since the model needs to be in Jy/pixel, it is necessary to reduce
    # the brightness as 1/f**2 to take account of the larger pixel size at
    # higher frequencies.
    sky_model_pbcor = cfg.sky_model.image.stem
    sky_model_pbcor += f"_pbcor_scan{n_scan_str}.fits"
    sky_model_pbcor = output_dcy / sky_model_pbcor
    pb_multiply(in_image=sky_model_mir_im, pb=scan_beam_fits,
                out_fitsfile=sky_model_pbcor, cellscal='1/F')
    sky_model_pbcor_mirim = sky_model_pbcor.with_suffix('.mirim')

    scan_out_ms: pathlib.Path = root_name.append(
        f'_ICUT_{n_scan_str}.ms'
    )
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
    temp_scan_out_mirvis = output_dcy / f'temp_{grc(10)}.mirvis'
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
    miriad.fits(op='uvout', _in=scan_out_mirvis, out=scan_out_uvfits)
    shutil.rmtree(scan_out_mirvis)

    casa.tasks.importuvfits(vis=f"{scan_out_ms}",
                            fitsfile=f"{scan_out_uvfits}")
    scan_out_uvfits.unlink()
    casa.tasks.vishead(vis=f"{scan_out_ms}", mode="put", hdkey="telescope",
                       hdvalue="SKA1-LOW")

    observation.products[scan]['MS'] = scan_out_ms

out_ms = root_name.append('.ms')
observation.concat_scan_measurement_sets(out_ms)

# ######################################################################## #
# ##################### Deconvolution/imaging run ######################## #
# ######################################################################## #
gaussian_taper_arcsec = 60.  # in arcsec
niter = 10000

LOGGER.info(f"Conducting deconvolution/imaging with WSClean")
wsclean_args = {
    'weight': 'uniform',
    'taper-gaussian': f'{gaussian_taper_arcsec}asec',
    'super-weight': 4,
    'name': f'{root_name}_wsclean',
    'size': f'{field.nx} {field.ny}',
    'scale': f'{field.cdelt * 3600}asec',
    'channels-out': correlator.n_chan,
    'niter': niter,
    'pol': 'xx'
}
farm.software.wsclean(f"{root_name}.ms", wsclean_args,
                      consolidate_channels=True)

# ######################################################################## #
# ########################### Finishing up ############################### #
# ######################################################################## #
pipeline_duration = (datetime.now() - pipeline_start)
LOGGER.info(f"Finished {pathlib.Path(__file__).name} pipeline in "
            f"{misc.timedelta_to_ddhhmmss(pipeline_duration)}")

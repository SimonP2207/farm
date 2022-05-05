import shutil
import copy
import sys
import logging
import argparse
import pathlib
import subprocess
from datetime import datetime

from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.time import TimeDelta
import astropy.units as u

import farm
import farm.data.loader as loader
import farm.physics.astronomy as ast
import farm.miscellaneous.error_handling as errh
import farm.miscellaneous.plotting as plotting
import farm.sky_model.tb_functions as tb_funcs
from farm.software.miriad import miriad
from farm.software import oskar
from farm.software.oskar import set_oskar_sim_beam_pattern, set_oskar_sim_interferometer
from farm.software.oskar import run_oskar_sim_beam_pattern, run_oskar_sim_interferometer
from farm import LOGGER

if __name__ == '__main__':
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_file",
                            help="Full path to farm configuration .toml file",
                            type=str)
        args = parser.parse_args()
        config_file = pathlib.Path(args.config_file)

    else:
        config_file = pathlib.Path(farm.data.FILES['EXAMPLE_CONFIG'])
# ############################################################################ #
# ###################### PARSE CONFIGURATION ################################# #
# ############################################################################ #
    cfg = loader.FarmConfiguration(config_file)
# ############################################################################ #
# ######################## SET UP THE LOGGER ################################# #
# ############################################################################ #
    now = datetime.now()
    logfile = f'farm{now.strftime("%Y%b%d_%H%M%S").upper()}.log'
    logfile = cfg.output_dcy / logfile
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
    console_handler.setLevel(logging.INFO)  # TODO: Parse this from a cl arg
    LOGGER.addHandler(console_handler)
# ############################################################################ #
# ########################### DEFINE VARIOUS FILE NAMES ###################### #
# TODO: Clean this section up and place in farm.loader.FarmConfiguration class #
# ############################################################################ #
    if not cfg.output_dcy.exists():
        LOGGER.info(f"Creating output directory, {cfg.output_dcy}")
        cfg.output_dcy.mkdir(exist_ok=True)
    # Define output file names for sky-models, images, and oskar config files
    # GDSM
    gsm = cfg.output_dcy.joinpath(f'{cfg.root_name}_GSM')
    gsmx = cfg.output_dcy.joinpath(f'{cfg.root_name}_GSMX')
    gsmf = cfg.output_dcy.joinpath(f'{cfg.root_name}_GSM.fits')

    # GDSM (high resolution)
    gtd = cfg.output_dcy.joinpath(f'{cfg.root_name}_GTD')
    gtdf = cfg.output_dcy.joinpath(f'{cfg.root_name}_GTD.fits')

    # Compact sky model (T-RECS)
    cmpt = cfg.output_dcy.joinpath(f'{cfg.root_name}_CMPT')

    # Define OSKAR specific names
    sbeam_ini = cfg.output_dcy.joinpath(f'{cfg.root_name}.ini')
    sbeam_name = cfg.output_dcy.joinpath(f'{cfg.root_name}_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I')
    sbeam_fname = sbeam_name.parent / f'{sbeam_name.name}.fits'

# ############################################################################ #
# ###################### Set up SkyModel instance ############################ #
# ############################################################################ #
    LOGGER.info(f"Instantiating SkyModel")
    sky_model = farm.sky_model.SkyModel((cfg.field.nx, cfg.field.ny),
                                        cfg.field.cdelt, cfg.field.coord0,
                                        cfg.correlator.frequencies)
    components = []
# ############################################################################ #
# ############ Determine scan times and produce diagnostic plots ############# #
# ############################################################################ #
    LOGGER.info("Computing scan times")
    # Compute scan times
    scan_times = ast.scan_times(
        cfg.observation.time, cfg.field.coord0, cfg.telescope.location,
        cfg.observation.n_scan, cfg.observation.t_total,
        cfg.observation.min_elevation, cfg.observation.min_gap_scan,
        partial_scans_allowed=False
    )

    # Round end times to sensible values consistent with visibility integration
    # times
    for i, (t_scan_start, t_scan_end) in enumerate(scan_times):
        n_ints = int((t_scan_end - t_scan_start).to_value('s') //
                     cfg.correlator.t_int)
        scan_duration = n_ints * cfg.correlator.t_int
        t_scan_end = t_scan_start + TimeDelta(scan_duration * u.s)
        scan_times[i] = (t_scan_start, t_scan_end)

    msg = 'Computed scan times are:\n'
    for idx, (start, end) in enumerate(scan_times):
        msg += f"Scan {idx + 1}: {start.strftime('%d%b%Y %H:%M:%S').upper()} " \
               f"to {end.strftime('%d%b%Y %H:%M:%S').upper()} " \
               f"({(end - start).to_value('s'):.1f}s)\n"
    LOGGER.info(msg)

    # Produce plot of elevation curve and proposed scans
    elevation_plot = cfg.output_dcy / "elevation_curve.pdf"
    LOGGER.info(f"Saving elevation plot to {elevation_plot}")
    plotting.target_altaz(
        cfg.observation.time, cfg.telescope.location,
        cfg.field.coord0, scan_times=scan_times,
        savefig=elevation_plot
    )
# ############################################################################ #
# ###################### TEC/Ionospheric calibration/effect ################## #
# TODO: This is where we left off on Friday 22/04/2022. Write code here to
#  produce TEC screen with ARatmospy, or load it from .fits
    if cfg.calibration.tec:
        if cfg.calibration.tec.create:
            LOGGER.info("Creating TEC screens from scratch")
            iono_root = cfg.output_dcy / 'iono_tec_'
            for i, (t_scan_start, t_scan_end) in tqdm(enumerate(scan_times),
                                                      desc='Creating TEC'):
                LOGGER.info(f"Creating TEC screen for scan {i + 1}")
                duration = (t_scan_end - t_scan_start).to_value('s')
                tec_fitsfile = iono_root.append(
                    str(i).zfill(len(str(len(scan_times)))) + '.fits'
                )
                farm.calibration.tec.create_tec_screen(
                    tec_fitsfile, np.mean(cfg.correlator.frequencies), 20., 20e3,
                    cfg.correlator.t_int, duration, cfg.calibration.noise.seed
                )
                cfg.calibration.tec.image.append(tec_fitsfile)
                LOGGER.info(f"TEC screen for scan {i + 1} saved to {tec_fitsfile}")
        else:
            # TODO: Code here to ensure that TEC images match up with desired scan times
            def check_tec_image_compatibility(configuration, tec_images):
                pass
            LOGGER.info("Checking compatibility of TEC .fits images with scans")
            check_tec_image_compatibility(cfg, cfg.calibration.tec.image)
    else:
        LOGGER.info("Not including TEC")
    print('help')
# ############################################################################ #
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Initial set up of OSKAR ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
    LOGGER.info("Setting up Oskar Sky instances")
    sky_fov = oskar.Sky()
    sky_side_lobes = oskar.Sky()
# ############################################################################ #
# ###################### Calculate station beams with OSKAR ################## #
# ############################################################################ #
    LOGGER.info("Setting up station beam pattern .ini files")
    with open(sbeam_ini, 'wt') as f:
        set_oskar_sim_beam_pattern(f, "simulator/double_precision", False)
        set_oskar_sim_beam_pattern(f, "observation/phase_centre_ra_deg",
                                   cfg.field.coord0.ra.deg)
        set_oskar_sim_beam_pattern(f, "observation/phase_centre_dec_deg",
                                   cfg.field.coord0.dec.deg)
        set_oskar_sim_beam_pattern(f, "observation/start_frequency_hz",
                                   cfg.correlator.freq_min)
        set_oskar_sim_beam_pattern(f, "observation/num_channels",
                                   cfg.correlator.n_chan)
        set_oskar_sim_beam_pattern(f, "observation/frequency_inc_hz",
                                   cfg.correlator.freq_inc)
        # set_oskar_sim_beam_pattern(f, "observation/start_time_utc",
        #                            ast.get_start_time(cfg.field.coord0.ra.deg,
        #                                               cfg.observation.t_total))
        # set_oskar_sim_beam_pattern(f, "observation/length",
        #                            cfg.observation.duration)
        # set_oskar_sim_beam_pattern(f, "observation/num_time_steps",
        #                            cfg.observation.n_scan)
        set_oskar_sim_beam_pattern(f, "observation/num_time_steps", 1)
        set_oskar_sim_beam_pattern(f, "telescope/input_directory",
                                   cfg.telescope.model)
        set_oskar_sim_beam_pattern(f, "telescope/pol_mode",
                                   "Scalar")
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/fov_deg",
                                   cfg.field.fov[0])
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/size",
                                   cfg.field.nx)
        # set_oskar_sim_beam_pattern(f, "beam_pattern/root_path",
        #                            cfg.root_name)
        set_oskar_sim_beam_pattern(f,
                                   "beam_pattern/station_outputs/fits_image/auto_power",
                                   True)

# ############################################################################ #
# #################### Calculate telescope model with OSKAR ################## #
# ############################################################################ #
    LOGGER.info("Setting up interferometer .ini files")
    sinterferometer_ini = pathlib.Path("test_sim_interferometer.ini")
    with open(sinterferometer_ini, 'wt') as f:
        set_oskar_sim_interferometer(f, 'simulator/double_precision', 'TRUE')
        set_oskar_sim_interferometer(f, 'simulator/use_gpus', 'FALSE')
        set_oskar_sim_interferometer(f, 'simulator/max_sources_per_chunk',
                                     '4096')

        set_oskar_sim_interferometer(f, 'observation/phase_centre_ra_deg',
                                     cfg.field.coord0.ra.deg)
        set_oskar_sim_interferometer(f, 'observation/phase_centre_dec_deg',
                                     cfg.field.coord0.dec.deg)
        set_oskar_sim_interferometer(f, 'observation/start_frequency_hz',
                                     cfg.correlator.freq_min)
        set_oskar_sim_interferometer(f, 'observation/num_channels',
                                     cfg.correlator.n_chan)
        set_oskar_sim_interferometer(f, 'observation/frequency_inc_hz',
                                     cfg.correlator.freq_inc)
        set_oskar_sim_interferometer(f, 'telescope/input_directory',
                                     cfg.telescope.model)
        set_oskar_sim_interferometer(f,
                                     'telescope/allow_station_beam_duplication',
                                     'TRUE')
        set_oskar_sim_interferometer(f, 'telescope/pol_mode', 'Scalar')
        # Add in ionospheric screen model
        set_oskar_sim_interferometer(f, 'telescope/ionosphere_screen_type',
                                     'External')
        set_oskar_sim_interferometer(f, 'interferometer/channel_bandwidth_hz',
                                     cfg.correlator.chan_width)
        set_oskar_sim_interferometer(f, 'interferometer/time_average_sec',
                                     cfg.correlator.t_int)
        set_oskar_sim_interferometer(f, 'interferometer/ignore_w_components',
                                     'FALSE')

        # Add in Telescope noise model via files where rms has been tuned
        set_oskar_sim_interferometer(f, 'interferometer/noise/enable',
                                     True if cfg.calibration.noise else False)
        set_oskar_sim_interferometer(f, 'interferometer/noise/seed',
                                     cfg.calibration.noise.seed)
        set_oskar_sim_interferometer(f, 'interferometer/noise/freq', 'Data')
        set_oskar_sim_interferometer(f, 'interferometer/noise/freq/file',
                                     cfg.calibration.noise.sefd_freq_file)
        set_oskar_sim_interferometer(f, 'interferometer/noise/rms', 'Data')
        set_oskar_sim_interferometer(f, 'interferometer/noise/rms/file',
                                     cfg.calibration.noise.sefd_rms_file)
        set_oskar_sim_interferometer(f, 'sky/fits_image/file',
                                     cfg.sky_model.image)
        set_oskar_sim_interferometer(f, 'sky/fits_image/default_map_units',
                                     'K')
# ############################################################################ #
# ####################### Large-scale foreground model ####################### #
# ############################################################################ #
    gdsm = None
    if cfg.sky_model.gdsm:
        if cfg.sky_model.gdsm.create:
            LOGGER.info("Creating large-scale Galactic foreground images")
            gdsm = farm.sky_model.SkyComponent(
                name='GDSM',
                npix=(cfg.field.nx, cfg.field.ny),
                cdelt=cfg.field.cdelt,
                coord0=cfg.field.coord0,
                tb_func=tb_funcs.gdsm2016_t_b
            )
            gdsm.add_frequency(cfg.correlator.frequencies)
        else:
            errh.raise_error(ValueError,
                             "Loading GDSM from image not currently supported")
            # if not cfg.sky_model.gdsm.image.exists():
            #     raise FileNotFoundError("Check path for GDSM image")
            # gdsm = farm.sky_model.SkyComponent.load_from_fits(
            #     fitsfile=cfg.sky_model.gdsm.image,
            #     name='GDSM'
            # )
        components.append(gdsm)
    else:
        LOGGER.info("Not including large-scale Galactic component")

# ############################################################################ #
# ##################### Small-scale foreground model ######################### #
# ############################################################################ #
    gssm = None
    if cfg.sky_model.gssm:
        fits_gssm = None
        if cfg.sky_model.gssm.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load GSSM model from fits")
        else:
            if cfg.sky_model.gssm.image == "":
                fits_gssm = farm.data.FILES['IMAGES']['MHD']
            elif cfg.sky_model.gssm.image.exists():
                fits_gssm = cfg.sky_model.gssm.image
            else:
                errh.raise_error(FileNotFoundError,
                                 f"{cfg.sky_model.gssm.image} does not exist")
            LOGGER.info(f"Loading Galactic small-scale component from "
                        f"{fits_gssm}")

        with fits.open(fits_gssm) as hdulist:
            fits_nx = np.shape(hdulist[0].data)[-1]

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

        gssm.cdelt *= 1.05
        LOGGER.info("Regridding small-scale images to large-scale parent")
        gssm = gssm.regrid(gdsm)

        LOGGER.info("Normalising small/large-scale power spectra")
        gssm.normalise(gdsm, inplace=True)
        components.append(gssm)
        merged = gssm.merge(gdsm, gssm.cdelt, 56. / 60., 'GASM')
        merged.write_fits(cfg.output_dcy / 'immerged.fits', 'JY/PIXEL')
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ######################## Extragalactic foreground model #################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ############################ A-Team sources ################################ #
# ############################################################################ #
    if cfg.sky_model.ateam:
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
    ps = None
    if cfg.sky_model.point_sources:
        LOGGER.info("Incorporating known point sources into foreground model")
        if cfg.sky_model.point_sources.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load model from fits")
        else:
            if cfg.sky_model.point_sources.image == "":
                # Parse .fits table data
                # TODO: Conditional here as to whether to load from the fits
                #  image or table
                LOGGER.info("Loading GLEAM catalogue")
                catalogue = farm.data.FILES['TABLES']['GLEAM']
                data = farm.data.fits_table_to_dataframe(catalogue)

                # Column name translation of GLEAM_EGC_v2.fits
                sky_model_cols = {'ra': 'RAJ2000', 'dec': 'DEJ2000',
                                  'fluxI': 'int_flux_wide',# 'fluxQ': None,
                                  #'fluxU': None, 'fluxV': None,
                                  'freq0': 'freq0', 'spix': 'alpha', #'rm': None,
                                  'maj': 'a_wide', 'min': 'b_wide',
                                  'pa': 'pa_wide'}

                # Add needed columns to DataFrame
                # TODO: Put generic spectral index of -0.7 somewhere sensible
                data[sky_model_cols['spix']] = np.where(np.isnan(data.alpha),
                                                        -0.7, data.alpha)
                data['freq0'] = 200e6  # GLEAM reference frequency
                data['_fov'] = ast.within_square_fov(
                    cfg.field.fov, cfg.field.coord0.ra.deg,
                    cfg.field.coord0.dec.deg,
                    data[sky_model_cols['ra']], data[sky_model_cols['dec']]
                )

                mask_fov = (
                    data['_fov'] & (
                        data.int_flux_wide <
                        cfg.sky_model.point_sources.flux_inner
                    )
                )
                mask_side_lobes = (data.int_flux_wide >
                                   cfg.sky_model.point_sources.flux_outer)

                # Add to fov Sky instance
                oskar.add_dataframe_to_sky(
                    data[mask_fov], sky_fov, sky_model_cols
                )

                # Add to side-lobes Sky instance
                oskar.add_dataframe_to_sky(
                    data[mask_side_lobes], sky_side_lobes, sky_model_cols
                )

                # Create SkyComponent instance and save .fits image of GLEAM
                # sources
                ps = farm.sky_model.SkyComponent.load_from_fits_table(
                    sky_model_cols, catalogue, 'GLEAM', cfg.field.cdelt,
                    cfg.field.coord0, fov=cfg.field.fov,
                    freqs=cfg.correlator.frequencies,
                    beam={'maj': 2. / 60, 'min': 2. / 60., 'pa': 0.}
                )

                ps.write_fits(pathlib.Path(f"{ps.name}_component.fits"),
                              unit='JY/PIXEL')

            elif not cfg.sky_model.point_sources.image.exists():
                errh.raise_error(FileNotFoundError,
                                 f"{str(cfg.sky_model.point_sources.image)} does not exist")
            else:
                errh.raise_error(NotImplementedError,
                                 "Currently can only load GLEAM model from "
                                 "fits table")
    else:
        LOGGER.info("Not including known point sources into foreground")
# ############################################################################ #
# ###################### Simulated sources from T-RECs ####################### #
# ############################################################################ #
    trecs_sources = None
    if cfg.sky_model.trecs:
        LOGGER.info("Incorporating T-RECS artificial sources into low-SNR "
                    "extragalactic foreground component")
        fits_trecs = None
        if cfg.sky_model.trecs.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load model from fits")
        else:
            if cfg.sky_model.trecs.image == "":
                fits_trecs = farm.data.FILES['IMAGES']['TRECS']
            elif cfg.sky_model.trecs.image.exists():
                fits_trecs = cfg.sky_model.trecs.image
            else:
                errh.raise_error(FileNotFoundError,
                                 f"{str(cfg.sky_model.trecs.image)} "
                                 "does not exist")
        # Parse .fits table data
        # TODO: Conditional here as to whether to load from the fits
        #  image or table
        # Create SkyComponent instance and save .fits image of GLEAM
        # sources
        trecs = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_trecs,
            name='TRECS',
            cdelt=cfg.field.cdelt,
            coord0=cfg.field.coord0,
            freqs=cfg.correlator.frequencies
        )

        trecs = trecs.regrid(gdsm)
        components.append(trecs)
    else:
        LOGGER.info("Not including T-RECS sources into foreground")
# ############################################################################ #
# ############################# EoR H-21cm signal ############################ #
# ############################################################################ #
    h21cm = None
    if cfg.sky_model.h21cm:
        LOGGER.info("Including EoR 21cm signal into sky model")
        fits_h21cm = None
        if cfg.sky_model.h21cm.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load model from fits")
        else:
            if cfg.sky_model.point_sources.image == "":
                fits_h21cm = farm.data.FILES['IMAGES']['H21CM']
            elif cfg.sky_model.h21cm.image.exists():
                fits_h21cm = cfg.sky_model.h21cm.image
            else:
                errh.raise_error(FileNotFoundError,
                                 f"{str(cfg.sky_model.h21cm.image)} "
                                 "does not exist")
        LOGGER.info(f"Loading EoR 21cm component from {fits_h21cm}")
        # Parse .fits table data
        # TODO: Conditional here as to whether to load from the fits
        #  image or table
        # Create SkyComponent instance and save .fits image of GLEAM
        # sources
        h21cm = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_h21cm,
            name='H21CM',
            cdelt=cfg.field.cdelt,
            coord0=cfg.field.coord0,
            freqs=cfg.correlator.frequencies
        )

        h21cm = h21cm.regrid(gdsm)
        components.append(h21cm)
    else:
        LOGGER.info("Not including EoR 21cm signal into sky model")
# ############################################################################ #
# ######## Add all components derived from .fits images to SkyModel, ######### #
# ###### write .fits images for image-derived components, and combine ######## #
# ############### side-lobe and fov Oskar.Sky instances ###################### #
# ############################################################################ #
    LOGGER.info("Adding all included sky components into sky model")
    sky_model += components
    for component in sky_model.components:
        component.write_fits(
            cfg.output_dcy / f"{component.name}_component.fits", unit='JY/PIXEL'
        )
    sky_model.write_fits(cfg.sky_model.image, unit='JY/PIXEL')
    sky_model_mir_im = cfg.sky_model.image.with_suffix('.im')
    sky_model.write_miriad_image(sky_model_mir_im, unit='JY/PIXEL')

    # Do not use copy.deepcopy on Sky instances. Throws the error:
    #     TypeError: unsupported operand type(s) for +: 'Sky' and 'int'
    sky = sky_fov.create_copy()
    sky.append(sky_side_lobes)
    LOGGER.info(f"Saving oskar sky model to {cfg.oskar_sky_model_file}")
    sky.save(str(cfg.oskar_sky_model_file))

    with open(sinterferometer_ini, 'at') as f:
        set_oskar_sim_interferometer(
            f, 'sky/oskar_sky_model/file', cfg.oskar_sky_model_file
        )

# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Synthetic observing run ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
    sbeam_sfx = '_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I'
    # sbeam_name = cfg.root_name.append(sbeam_sfx)
    # sbeam_fname = sbeam_name.append('.fits')

    out_ms = cfg.root_name / '.ms'
    out_msm = cfg.root_name / '.msm'

    LOGGER.info("Running synthetic observations")
    for icut, (t_scan_start, t_scan_end) in enumerate(scan_times):
        t_scan = (t_scan_end - t_scan_start).to_value('s')
        sbeam_root = cfg.root_name.append(f"_scan{icut}")
        sbeam_name = sbeam_root.append(sbeam_sfx)
        sbeam_fname = sbeam_name.append('.fits')

        # Set up beam-pattern for scan
        LOGGER.info(f"Running oskar_sim_beam_pattern from {sbeam_ini}")
        with open(sbeam_ini, 'at') as f:
            set_oskar_sim_beam_pattern(
                f, "beam_pattern/root_path", sbeam_root
            )
            set_oskar_sim_beam_pattern(f, "observation/start_time_utc",
                                       t_scan_start)
            set_oskar_sim_beam_pattern(f, "observation/length", t_scan)
        run_oskar_sim_beam_pattern(sbeam_ini)
        sbeam_hdu = fits.open(sbeam_fname)  #

        LOGGER.info(f"Starting synthetic observations' scan #{icut + 1}")
        # TODO: End of 27APR22. Figure out sbmout etc below. I think we don't
        #  need a lot of the files as each scan's beam has only one 'frame' in
        #  time in the beam cube
        sicut = str(icut).zfill(len(str(cfg.observation.n_scan)) + 1)
        sbmdata_out = sbeam_hdu[0].data[0, :, :, :]
        sbmout_hdu = fits.PrimaryHDU(sbmdata_out)
        sbmout_cut = cfg.root_name.append('_ICUT_' + sicut)
        sbmout_fcut = cfg.root_name.append('_ICUT_' + sicut + '.fits')

        # TODO: This is where we left off. Correct below to be farm-compatible
        #  code
        gsm_cut = gsm.append('_ICUT_' + sicut)
        gsm_pcut = gsm.append('_pICUT_' + sicut)
        gsm_tcut = gsm.append('_tICUT_' + sicut)

        ionof_cut = cfg.calibration.tec.image[icut]
        ionot_cut = cfg.root_name.append('_iIcut_' + sicut)  # TEC screen for this time cut (miriad image)
        ionom_cut = cfg.root_name.append('_imIcut_' + sicut)  # TEC screen + residual errors for this time cut (miriad image)
        iono_cut = cfg.root_name.append('_iIcut_' + sicut + '.fits')  # TEC screen + residual errors for this time cut (.fits image)

        cmpt_mscut = cmpt.append('_ICUT_' + sicut + '.ms')
        cmpt_msmcut = cmpt.append('_ICUT_' + sicut + '.msm')
        cmpt_uvfcut = cmpt.append('_ICUT_' + sicut + '.uvf')
        cmpt_uvcut = cmpt.append('_ICUT_' + sicut + '.uv')
        out_uvcut = cfg.root_name.append('_ICUT_' + sicut + '.uv')
        out_mscut = cfg.root_name.append('_ICUT_' + sicut + '.ms')
        out_uvfcut = cfg.root_name.append('_ICUT_' + sicut + '.uvf')

        sbmout_hdu.header.set('CTYPE1', 'RA---SIN')
        sbmout_hdu.header.set('CTYPE2', 'DEC--SIN')
        sbmout_hdu.header.set('CTYPE3', 'FREQ    ')
        sbmout_hdu.header.set('CRVAL1', cfg.field.coord0.ra.deg)
        sbmout_hdu.header.set('CRVAL2', cfg.field.coord0.dec.deg)
        sbmout_hdu.header.set('CRVAL3', cfg.correlator.freq_min)
        sbmout_hdu.header.set('CRPIX1', cfg.field.nx // 2)
        sbmout_hdu.header.set('CRPIX2', cfg.field.ny // 2)
        sbmout_hdu.header.set('CRPIX3', 1)
        sbmout_hdu.header.set('CDELT1', -cfg.field.cdelt)
        sbmout_hdu.header.set('CDELT2', cfg.field.cdelt)
        sbmout_hdu.header.set('CDELT3', cfg.correlator.freq_inc)
        sbmout_hdu.header.set('CUNIT1', 'deg     ')
        sbmout_hdu.header.set('CUNIT2', 'deg     ')
        sbmout_hdu.header.set('CUNIT3', 'Hz      ')
        sbmout_hdu.writeto(sbmout_fcut, overwrite=True)

        LOGGER.info(f"Multiplying sky model by beam response, {sbmout_cut}")
        if sbmout_cut.exists():
            shutil.rmtree(sbmout_cut)
        miriad.fits(op="xyin", _in=sbmout_fcut, out=sbmout_cut)
        miriad.maths(exp=f"<{sbmout_cut}>*<{sky_model_mir_im}>", out=gsm_pcut)

        text = f'/bin/cp -r {gsm_pcut} {gsm_tcut}'
        subprocess.run(text, shell='True')
        miriad.puthd(_in=f"{gsm_tcut}/cellscal", value="1/F")
        miriad.regrid(_in=gsm_pcut, tin=gsm_tcut, out=gsm_cut)

        # Get the relevant cut from the ionospheric model and scale for net
        # residual effect
        miriad.fits(op="xyin", _in=ionof_cut, out=ionot_cut)
        miriad.maths(exp=f"<{ionot_cut}>*{cfg.calibration.tec.err:.3e}",
                     out=ionom_cut)
        miriad.fits(op="xyout", _in=ionom_cut, out=iono_cut)

        # Create measurement sets
        with open(sinterferometer_ini, 'at') as f:
            set_oskar_sim_interferometer(
                f, 'observation/start_time_utc',
                t_scan_start.strftime("%Y/%m/%d/%H:%M:%S.%f")[:-2]
            )
            set_oskar_sim_interferometer(
                f, 'telescope/external_tec_screen/input_fits_file', iono_cut
            )
            set_oskar_sim_interferometer(
                f, 'interferometer/ms_filename', cmpt_mscut
            )

            set_oskar_sim_interferometer(
                f, 'observation/length', format(t_scan, '.1f')
            )
            set_oskar_sim_interferometer(
                f, 'observation/num_time_steps',
                int(t_scan // cfg.correlator.t_int)
            )
        run_oskar_sim_interferometer(sinterferometer_ini)

        script_line0 = f'vishead(vis="{cmpt_mscut}", mode="put", hdkey="telescope", hdvalue="SKA1-LOW")\n'
        script_line1 = f'exportuvfits(vis="{cmpt_mscut}", fitsfile="{cmpt_uvfcut}", datacolumn="data", multisource=False, writestation=False, overwrite=True)\n'
        with open('_casa_script.py', 'w') as f:
            f.write(script_line0)
            f.write(script_line1)

        text = f"{farm.software.which('casa')} --nologger -c _casa_script.py"
        subprocess.run(text, shell='True')

        # TODO: Script fails here because of the modifications to miriad input
        #  args/kwargs made today. Need another way around the miriad character
        #  limit that doesn't include recompiling miriad! Perhaps all miriad
        #  tasks executed should also execute with chdir commands and just filename
        #  inputs?
        miriad.fits(op='uvin', _in=cmpt_uvfcut, options="nofq", out=cmpt_uvcut)
        miriad.uvmodel(vis=cmpt_uvcut, model=gsm_cut, options="add,zero",
                       out=out_uvcut)
        miriad.gperror(vis=out_uvcut, interval=1.,
                       pnoise=cfg.calibration.gains.phase_err,
                       gnoise=cfg.calibration.gains.amp_err)
        miriad.fits(op='uvout', _in=out_uvcut, out=out_uvfcut)

        script_line0 = f'importuvfits(vis={out_mscut}, fitsfile={out_uvfcut})\n'
        with open('_casa_script.py', 'w') as f:
            f.write(script_line0)

        text = f"{farm.software.which('casa')} --nologger -c _casa_script.py"
        subprocess.run(text, shell='True')

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

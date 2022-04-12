import sys
import logging
import argparse
import pathlib
from typing import Union
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.io import fits

import farm
import farm.data.loader as loader
import farm.physics.astronomy as ast
import farm.miscellaneous.error_handling as errh
import farm.sky_model.tb_functions as tb_funcs
from farm.software import oskar
from farm.software.oskar import set_oskar_sim_beam_pattern, set_oskar_sim_interferometer
from farm.software.oskar import run_oskar_sim_beam_pattern
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
# ########################### DEFINE VARIOUS FILE NAMES ###################### #
# TODO: Clean this section up and place in farm.loader.FarmConfiguration class #
# ############################################################################ #
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
# ######################## SET UP THE LOGGER ################################# #
# ############################################################################ #
    now = datetime.now()
    logfile = f'farm{now.strftime("%Y%b%d_%H%M%S").upper()}.log'
    logfile = cfg.output_dcy / logfile
    LOGGER.setLevel(logging.DEBUG)  # TODO: Parse this from a command-line arg
    fh = logging.FileHandler(str(logfile), mode="w",
                             encoding=sys.stdout.encoding)
    fh.setFormatter(logging.Formatter(farm.LOG_FMT, datefmt=farm.LOG_DATE_FMT))
    LOGGER.addHandler(fh)
# ############################################################################ #
# ###################### Set up SkyModel instance ############################ #
# ############################################################################ #
    sky_model = farm.sky_model.SkyModel((cfg.field.nx, cfg.field.ny),
                                        cfg.field.cdelt, cfg.field.coord0,
                                        cfg.correlator.frequencies)
    components = []
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Initial set up of OSKAR ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
    sky_fov = oskar.Sky()
    sky_side_lobes = oskar.Sky()
# ############################################################################ #
# ###################### Calculate station beams with OSKAR ################## #
# ############################################################################ #
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
        set_oskar_sim_beam_pattern(f, "observation/start_time_utc",
                                   ast.get_start_time(cfg.field.coord0.ra.deg,
                                                      cfg.observation.duration))
        set_oskar_sim_beam_pattern(f, "observation/length",
                                   cfg.observation.duration)
        set_oskar_sim_beam_pattern(f, "observation/num_time_steps",
                                   cfg.observation.n_scan)
        set_oskar_sim_beam_pattern(f, "telescope/input_directory",
                                   cfg.telescope.model)
        set_oskar_sim_beam_pattern(f, "telescope/pol_mode",
                                   "Scalar")
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/fov_deg",
                                   cfg.field.fov[0])
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/size",
                                   cfg.field.nx)
        set_oskar_sim_beam_pattern(f, "beam_pattern/root_path",
                                   cfg.root_name)
        set_oskar_sim_beam_pattern(f,
                                   "beam_pattern/station_outputs/fits_image/auto_power",
                                   True)
# ############################################################################ #
# #################### Calculate telescope model with OSKAR ################## #
# ############################################################################ #
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
        set_oskar_sim_interferometer(f, 'observation/length',
                                     cfg.observation.t_scan)
        set_oskar_sim_interferometer(f, 'observation/num_time_steps',
                                     int(cfg.observation.t_scan //
                                         cfg.correlator.t_int))
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
                                     cfg.calibration.noise)
        set_oskar_sim_interferometer(f, 'interferometer/noise/seed',
                                     cfg.calibration.noise_seed)
        set_oskar_sim_interferometer(f, 'interferometer/noise/freq', 'Data')
        set_oskar_sim_interferometer(f, 'interferometer/noise/freq/file',
                                     cfg.calibration.sefd_freq_file)
        set_oskar_sim_interferometer(f, 'interferometer/noise/rms', 'Data')
        set_oskar_sim_interferometer(f, 'interferometer/noise/rms/file',
                                     cfg.calibration.sefd_rms_file)
        set_oskar_sim_interferometer(f, 'sky/fits_image/file',
                                     '/Users/simon.purser/pylib/farm/test_skymodel.fits')
        set_oskar_sim_interferometer(f, 'sky/fits_image/default_map_units',
                                     'K')
# ############################################################################ #
# ####################### Large-scale foreground model ####################### #
# ############################################################################ #
    gdsm = None
    if cfg.sky_model.gdsm:
        if cfg.sky_model.gdsm.create:
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
        gssm = farm.sky_model.SkyComponent.load_from_fits(
            fitsfile=fits_gssm,
            name='GSSM',
            cdelt=cfg.field.fov[0] / 512,
            coord0=cfg.field.coord0,
            freqs=cfg.correlator.frequencies
        )
        gssm.rotate(angle=ast.angle_to_galactic_plane(cfg.field.coord0),
                    inplace=True)
        gssm = gssm.regrid(gdsm)
        gssm.normalise(gdsm, inplace=True)
        components.append(gssm)
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ######################## Extragalactic foreground model #################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ############################ A-Team sources ################################ #
# ############################################################################ #
    if cfg.sky_model.ateam:
        ateam_data = farm.data.ATEAM_DATA
        sky_fov.append_sources(**ateam_data)

        # Imperfectly-demixed, residual A-Team sources in side-lobes
        if cfg.sky_model.ateam.demix_error:
            residual_fac = np.ones(len(ateam_data.columns))
            residual_fac.put(ateam_data.columns.get_loc('I'),
                             cfg.sky_model.ateam.demix_error)
            sky_side_lobes.append_sources(**ateam_data * residual_fac)
# ############################################################################ #
# ########################## Real sources from surveys ####################### #
# ############################################################################ #
    # Also, separate sky model for in-fov and out-fov sources with flux cutoff
    # for each
    ps = None
    if cfg.sky_model.point_sources:
        if cfg.sky_model.point_sources.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load model from fits")
        else:
            if cfg.sky_model.point_sources.image == "":
                # Parse .fits table data
                # TODO: Conditional here as to whether to load from the fits
                #  image or table
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
# ############################################################################ #
# ###################### Simulated sources from T-RECs ####################### #
# ############################################################################ #
    trecs_sources = None
    if cfg.sky_model.trecs:
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
# ############################################################################ #
# ############################# EoR H-21cm signal ############################ #
# ############################################################################ #
    h21cm = None
    if cfg.sky_model.h21cm:
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
# ############################################################################ #
# ######## Add all components derived from .fits images to SkyModel ########## #
# ########## and write .fits images for image-derived components ############# #
# ############################################################################ #
    sky_model += components
    for component in sky_model.components:
        component.write_fits(pathlib.Path(f"{component.name}_component.fits"),
                             unit='JY/PIXEL')
    sky_model.write_fits(pathlib.Path('test_skymodel.fits'), unit='JY/PIXEL')
# ############################################################################ #
# ########################  # # # # # # # # # # # # # # # #################### #
# ########################## Synthetic observing run ######################### #
# ########################  # # # # # # # # # # # # # # # #################### #
# ############################################################################ #
# ########## Create oskar.Sky instances for fov and side-lobes to ############ #
# ############# hold 'tabulated', Gaussian foreground sources ################ #
# ############################################################################ #
    run_oskar_sim_beam_pattern(sbeam_ini)
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

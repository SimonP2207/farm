import sys
import logging
import argparse
import pathlib
from datetime import datetime

import farm
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

import farm.data.loader as loader
import farm.astronomy as ast
import miscellaneous.error_handling as errh
import farm.tb_functions as tb_funcs
from farm.calibration.noise import sefd_to_rms
from farm.software import oskar
from farm.software.oskar import set_oskar_sim_beam_pattern
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
        config_file = pathlib.Path(farm.data.DATA_FILES['EXAMPLE_CONFIG'])
# ############################################################################ #
# ###################### PARSE CONFIGURATION ################################# #
# ############################################################################ #
    # cfg = loader.load_configuration(config_file)
    cfg = loader.FarmConfiguration(config_file)
    # Files/directories
    # tel_dir = pathlib.Path(cfg["directories"]["telescope_model"])
    # output_dir = pathlib.Path(cfg["directories"]["output_dcy"])
    # sefd_freq_file = pathlib.Path(cfg["calibration"]["noise"]["sefd_frequencies_file"])
    # sefd_file = pathlib.Path(cfg["calibration"]["noise"]["sefd_file"])
    # gdsm_file = pathlib.Path(cfg["sky_models"]["GDSM"]["image"])
    # gssm_file = pathlib.Path(cfg["sky_models"]["GSSM"]["image"])
    # points_sources_file = pathlib.Path(cfg["sky_models"]["PS"]["image"])
    # freq_min = cfg["observation"]["correlator"]["freq_min"]  # [Hz]
    # freq_max = cfg["observation"]["correlator"]["freq_max"]  # [Hz]
    # nchan = cfg["observation"]["correlator"]["nchan"]  # [Hz]
    # chan_inc = cfg["observation"]["correlator"]["chanwidth"]  # channel BW [Hz]
    # fov_deg = cfg["observation"]["field"]["fov"]  # diffuse sky model FOV [deg]
    # coord0 = SkyCoord(cfg["observation"]["field"]["ra0"],
    #                   cfg["observation"]["field"]["dec0"],
    #                   unit=(u.hourangle, u.degree),
    #                   frame=cfg["observation"]["field"]["frame"])
    # nx = cfg["observation"]["field"]["nxpix"]  # pixels in field
    # ny = cfg["observation"]["field"]["nypix"]  # pixels in field
    # ra0 = cfg["observation"]["field"]["ra0"]
    # dec0 = cfg["observation"]["field"]["dec0"]
    # frame = cfg["observation"]["field"]["frame"]
    # flux_val = cfg["sky_models"]["PS"]["flux_cutoff"]  # flux cutoff beyond radius of fov_deg [Jy]
    # length_sec = cfg["observation"]["duration"]  # total length of observation
    # cut_sec = cfg["observation"]["t_scan"]  # length of each cut ('scan')
    # int_sec = cfg["observation"]["correlator"]["t_int"]  # integration time
    # num_cuts = cfg["observation"]["n_scan"]  # number of cuts ('scans')
    # noise_seed = cfg["calibration"]['noise']["noise_seed"]
    # out_root = cfg['root_name']  # root name for all output files (not directory)
# ############################################################################ #
# ########################### DEFINE VARIOUS FILE NAMES ###################### #
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
    logfile = cfg.output_dcy.joinpath(f'farm{now.strftime("%d%b%Y_%H%M%S").upper()}.log')
    LOGGER.setLevel(logging.DEBUG)
    log_fmt = "%(asctime)s:: %(levelname)s:: %(module)s.%(funcName)s:: %(message)s"
    fh = logging.FileHandler(str(logfile), mode="w",
                             encoding=sys.stdout.encoding)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(fh)
# ############################################################################ #
# ###################### CONFIGURATION LOGIC ################################# #
# ############################################################################ #
#     cdelt = fov_deg / nx
#     freqs = np.linspace(freq_min, freq_max, nchan)
#     freq_inc = freqs[1] - freqs[0]
#     start_utc = ast.get_start_time(coord0.ra.deg, length_sec)
#     num_int_per_scan = cut_sec / int_sec

    # Calculate and save image rms levels from SEFDs
    # rms_file = cfg.output_dcy / "rms_noise_file.txt"
    # t_total = cut_sec * num_cuts
    # n_ants = len(np.loadtxt(tel_dir / 'layout.txt'))
    # im_rms = sefd_to_rms(np.loadtxt(sefd_file), n_ants, t_total, chan_inc)
    # np.savetxt(rms_file, im_rms)
    #
    sky_model = farm.sky_model.SkyModel((cfg.field.nx, cfg.field.ny),
                                        cfg.field.cdelt, cfg.field.coord0,
                                        cfg.correlator.frequencies)
    components = []
# ############################################################################ #
# ####################### Large-scale foreground model ####################### #
# ############################################################################ #
    gdsm = None
    if cfg.sky_model.gdsm:
        if cfg.sky_model.gdsm.create:
            gdsm = farm.sky_model.SkyComponent(
                'GDSM', (cfg.field.nx, cfg.field.ny), cdelt=cfg.field.cdelt,
                 coord0=cfg.field.coord0, tb_func=tb_funcs.gdsm2016_t_b
            )
            gdsm.add_frequency(cfg.correlator.frequencies)
        else:
            errh.raise_error(ValueError,
                             "Loading GDSM from image not currently supported")
            if not cfg.sky_model.gdsm.image.exists():
                raise FileNotFoundError("Check path for GDSM image")
            gdsm = farm.sky_model.SkyComponent.load_from_fits(cfg.sky_model.gdsm.image, 'GDSM')
        components.append(gdsm)
# ############################################################################ #
# ##################### Small-scale foreground model ######################### #
# ############################################################################ #
    gssm = None
    if cfg.sky_model.gssm:
        if cfg.sky_model.gssm.create:
            errh.raise_error(NotImplementedError,
                             "Currently can only load GSSM model from fits")
        else:
            if cfg.sky_model.gssm.image == "":
                gssm = farm.sky_model.SkyComponent.load_from_fits(
                    farm.data.DATA_FILES['MHD'], 'GSSM', cfg.field.fov[0] / 512,
                    cfg.field.coord0
                )
                if(len(cfg.correlator.frequencies) != len(gssm.frequencies) or
                   not all(np.isclose(cfg.correlator.frequencies, gssm.frequencies, atol=1.))):
                    raise ValueError("GSSM .fits cube frequencies differ from"
                                     "those requested")
                gssm.rotate(angle=ast.angle_to_galactic_plane(cfg.field.coord0),
                            inplace=True)
            elif not cfg.sky_model.gssm.image.exists():
                errh.raise_error(FileNotFoundError,
                                 f"{str(cfg.sky_model.gssm.image)} does not exist")
            else:
                gssm = farm.sky_model.SkyComponent.load_from_fits(cfg.sky_model.gssm.image, 'GSSM')
            gssm = gssm.regrid(gdsm)
            gssm.normalise(gdsm, inplace=True)
        components.append(gssm)
# ############################################################################ #
# ######################## TRECS foreground model ############################ #
# ############################################################################ #
    point_sources = None
    if cfg.sky_model.point_sources:
        if cfg["sky_models"]["PS"]["create"]:
            errh.raise_error(NotImplementedError,
                             "Currently can only load TRECS model from fits")
        else:
            if cfg.sky_model.point_sources.image == "":
                ps = farm.sky_model.SkyComponent.load_from_fits(
                    farm.data.DATA_FILES['PS'], 'PS', cfg.field.fov[0] / 512,
                    cfg.field.coord0
                )
                if(len(cfg.correlator.frequencies) != len(ps.frequencies) or
                   not all(np.isclose(cfg.correlator.frequencies, ps.frequencies, atol=1.))):
                    raise ValueError("Point sources .fits cube frequencies "
                                     "differ from those requested")
            elif not cfg.sky_model.point_sources.image.exists():
                errh.raise_error(FileNotFoundError,
                                 f"{str(cfg.sky_model.point_sources.image)} does not exist")
            else:
                point_sources = farm.sky_model.SkyComponent.load_from_fits(
                    cfg.sky_model.point_sources.image, 'PS'
                )
            point_sources = point_sources.regrid(gdsm)
# ############################################################################ #
# ############################ A-Team sources ################################ #
# ############################################################################ #
    if cfg.sky_model.ateam:
        if cfg.sky_model.ateam.create:
            sky0 = oskar.Sky()
            sky1 = oskar.Sky()
            sky_bright = oskar.Sky.from_array(farm.data.ATEAM_DATA)
            sky_bright_att = oskar.Sky.from_array(farm.data.ATEAM_DATA)
            sky0.append(sky_bright)
            sky1.append(sky_bright_att)
        else:
            raise(ValueError, "A-Team loading from file not yet supported")
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
                                   cfg.telescope_model)
        set_oskar_sim_beam_pattern(f, "telescope/pol_mode",
                                   "Scalar")
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/fov_deg",
                                   cfg.field.fov[0])
        set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/size",
                                   cfg.field.nx)
        set_oskar_sim_beam_pattern(f, "beam_pattern/root_path",
                                   cfg.root_name)
        set_oskar_sim_beam_pattern(f, "beam_pattern/station_outputs/fits_image/auto_power",
                                   True)
    run_oskar_sim_beam_pattern(sbeam_ini)
# ############################################################################ #
# #################### Calculate telescope model with OSKAR ################## #
# ############################################################################ #
    tscp_settings = oskar.SettingsTree('oskar_sim_interferometer')

    tscp_settings.set_value('simulator/double_precision', 'TRUE')
    tscp_settings.set_value('simulator/use_gpus', 'FALSE')
    tscp_settings.set_value('simulator/max_sources_per_chunk', '4096')

    tscp_settings.set_value('observation/phase_centre_ra_deg',
                            cfg.field.coord0.ra.deg)
    tscp_settings.set_value('observation/phase_centre_dec_deg',
                            cfg.field.coord0.dec.deg)
    tscp_settings.set_value('observation/start_frequency_hz',
                            cfg.correlator.freq_min)
    tscp_settings.set_value('observation/num_channels',
                            cfg.correlator.n_chan)
    tscp_settings.set_value('observation/frequency_inc_hz',
                            cfg.correlator.freq_inc)
    tscp_settings.set_value('observation/length',
                            cfg.observation.t_scan)
    tscp_settings.set_value('observation/num_time_steps',
                            int(cfg.observation.t_scan // cfg.correlator.t_int))

    tscp_settings.set_value('telescope/input_directory', cfg.telescope_model)
    tscp_settings.set_value('telescope/allow_station_beam_duplication', 'TRUE')
    tscp_settings.set_value('telescope/pol_mode', 'Scalar')
    # Add in ionospheric screen model
    tscp_settings.set_value('telescope/ionosphere_screen_type', 'External')

    tscp_settings.set_value('interferometer/channel_bandwidth_hz', cfg.correlator.chan_width)
    tscp_settings.set_value('interferometer/time_average_sec', cfg.correlator.t_int)
    tscp_settings.set_value('interferometer/ignore_w_components', 'FALSE')

    # Add in Telescope noise model via files where rms has been tuned
    tscp_settings.set_value('interferometer/noise/enable', cfg.calibration.noise)
    tscp_settings.set_value('interferometer/noise/seed', cfg.calibration.noise_seed)
    tscp_settings.set_value('interferometer/noise/freq', 'Data')
    tscp_settings.set_value('interferometer/noise/freq/file', cfg.calibration.sefd_freq_file)
    tscp_settings.set_value('interferometer/noise/rms', 'Data')
    tscp_settings.set_value('interferometer/noise/rms/file', cfg.calibration.sefd_rms_file)
# ############################################################################ #
# ############################################################################ #
# ############################################################################ #
    sky_model += components
    gdsm.write_fits(pathlib.Path('test_gdsm.fits'), unit='K')
    gssm.write_fits(pathlib.Path('test_gssm.fits'), unit='K')
    sky_model.write_fits(pathlib.Path('test_skymodel.fits'), unit='K')
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

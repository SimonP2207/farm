import sys
import logging

import pathlib
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

import loader
import classes
import astronomy as ast
from cli.oskar import set_oskar_sim_beam_pattern, run_oskar_sim_beam_pattern
from cli.miriad import miriad
from farm import LOGGER

# ############################################################################ #
# ###################### PARSE CONFIGURATION ################################# #
# ############################################################################ #
# Files/directories
cfg = loader.load_configuration("config.toml")
sm_dir = pathlib.Path(cfg["directories"]["sky_models"])
tel_dir = pathlib.Path(cfg["directories"]["telescope_model"])
output_dir = pathlib.Path(cfg["directories"]["output"])
gsmfile = pathlib.Path(cfg["sky_models"]["GDSM"]["image"])

freq_min = cfg["observation"]["correlator"]["freq_min"]  # in Hz
freq_max = cfg["observation"]["correlator"]["freq_max"]  # in Hz
nchan = cfg["observation"]["correlator"]["nchan"]  # in Hz
chan_inc = cfg["observation"]["correlator"]["chanwidth"]  # channel BW for smearing calculation in Hz
fov_deg = cfg["observation"]["field"]["fov"]  # in degrees for diffuse sky model
coord0 = SkyCoord(cfg["observation"]["field"]["ra0"],
                  cfg["observation"]["field"]["dec0"],
                  unit=(u.hourangle, u.degree),
                  frame=cfg["observation"]["field"]["frame"])
nx = cfg["sky_models"]["GDSM"]["nx"]  # pixels in diffuse sky model
ny = cfg["sky_models"]["GDSM"]["ny"]  # pixels in diffuse sky model
ra0 = cfg["observation"]["field"]["ra0"]
dec0 = cfg["observation"]["field"]["dec0"]
frame = cfg["observation"]["field"]["frame"]
flux_val = cfg["sky_models"]["PS"]["flux_cutoff"]  # flux cutoff beyond radius of fov_deg, Jy
length_sec = cfg["observation"]["duration"]  # total length of observation
cut_sec = cfg["observation"]["t_scan"]  # length of each cut ('scan')
int_sec = cfg["observation"]["correlator"]["t_int"]  # integration time
num_cuts = cfg["observation"]["n_scan"]  # number of cuts ('scans')

out_root = cfg['root_name']  # root name for all output files (not directory)
# ############################################################################ #
# ########################### DEFINE VARIOUS FILE NAMES ###################### #
# ############################################################################ #
output_dir.mkdir(exist_ok=True)

# Define output file names for sky-models, images, and oskar config files
# GDSM
gsm = output_dir.joinpath(f'{out_root}_GSM')
gsmx = output_dir.joinpath(f'{out_root}_GSMX')
gsmf = output_dir.joinpath(f'{out_root}_GSM.fits')

# GDSM (high resolution)
gtd = output_dir.joinpath(f'{out_root}_GTD')
gtdf = output_dir.joinpath(f'{out_root}_GTD.fits')

# Compact sky model (T-RECS)
cmpt = output_dir.joinpath(f'{out_root}_CMPT')

# Define OSKAR specific names
sbeam_ini = output_dir.joinpath(f'{out_root}.ini')
sbeam_name = output_dir.joinpath(f'{out_root}_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I')
sbeam_fname = output_dir.joinpath(f'{out_root}_S0000_TIME_SEP_CHAN_SEP_AUTO_POWER_AMP_I_I.fits')
# ############################################################################ #
# ######################## SET UP THE LOGGER ################################# #
# ############################################################################ #
now = datetime.now()
logfile = output_dir.joinpath(f'farm{now.strftime("%d%b%Y_%H%M%S").upper()}.log')
LOGGER.setLevel(logging.DEBUG)
log_fmt = "%(asctime)s:: %(levelname)s:: %(module)s.%(funcName)s:: %(message)s"
fh = logging.FileHandler(str(logfile), mode="w",
                         encoding=sys.stdout.encoding)
fh.setFormatter(logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
LOGGER.addHandler(fh)
# ############################################################################ #
# ###################### CONFIGURATION LOGIC ################################# #
# ############################################################################ #
coord0 = SkyCoord(ra0, dec0, unit=(u.hourangle, u.degree), frame=frame)
cdelt = fov_deg / nx
freqs = np.linspace(freq_min, freq_max, nchan)
freq_inc = freqs[1] - freqs[0]
# ############################################################################ #
# ###################### Calculate station beams with OSKAR ################## #
# ############################################################################ #
phi = ast.angle_to_galactic_plane(coord0)
start_utc = ast.get_start_time(coord0.ra.deg, length_sec)
with open(sbeam_ini, 'wt') as f:
    set_oskar_sim_beam_pattern(f, "simulator/double_precision", False)
    set_oskar_sim_beam_pattern(f, "observation/phase_centre_ra_deg",
                               coord0.ra.deg)
    set_oskar_sim_beam_pattern(f, "observation/phase_centre_dec_deg",
                               coord0.dec.deg)
    set_oskar_sim_beam_pattern(f, "observation/start_frequency_hz", freq_min)
    set_oskar_sim_beam_pattern(f, "observation/num_channels", nchan)
    set_oskar_sim_beam_pattern(f, "observation/frequency_inc_hz", freq_inc)
    set_oskar_sim_beam_pattern(f, "observation/start_time_utc",
                               ast.get_start_time(coord0.ra.deg, length_sec))
    set_oskar_sim_beam_pattern(f, "observation/length", length_sec)
    set_oskar_sim_beam_pattern(f, "observation/num_time_steps", num_cuts)
    set_oskar_sim_beam_pattern(f, "telescope/input_directory", tel_dir)
    set_oskar_sim_beam_pattern(f, "telescope/pol_mode", "Scalar")
    set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/fov_deg", fov_deg)
    set_oskar_sim_beam_pattern(f, "beam_pattern/beam_image/size", nx)
    set_oskar_sim_beam_pattern(f, "beam_pattern/root_path", out_root)
    set_oskar_sim_beam_pattern(f, "beam_pattern/station_outputs/fits_image/auto_power", True)
run_oskar_sim_beam_pattern(sbeam_ini)
# ############################################################################ #
# ####################### Large-scale foreground model ####################### #
# ############################################################################ #
if cfg["sky_models"]["GDSM"]["include"]:
    if cfg["sky_models"]["GDSM"]["create"]:
        gdsm = classes.DiffuseSkyModel((nx, ny), cdelt=cdelt,
                                       coord0=coord0, model='GSM2016')
        gdsm.add_frequency(freqs)
    else:
        raise ValueError("Loading GDSM from image not currently supported")
        if not gsmfile.exists():
            raise FileNotFoundError("Check path for GDSM image")
        gdsm = classes.DiffuseSkyModel.load_from_fits(gsmfile)
else:
    gdsm = None

gdsm.generate_fits(gsmf, unit='JY/PIXEL')
gdsm.generate_miriad_image(gsm, unit='JY/PIXEL')
# ############################################################################ #
# ##################### Small-scale foreground model ######################### #
# ############################################################################ #
gssm_fits = cfg["sky_models"]["GSSM"]["image"]
hdu_gssm = fits.open(gssm_fits)[0]
hdr_gssm, imdata_gssm = hdu_gssm.header, hdu_gssm.data
freqs_gssm = np.linspace(hdr_gssm['CRVAL3'] -
                         ((hdr_gssm['CRPIX3'] - 1) * hdr_gssm['CDELT3']),
                         hdr_gssm['CRVAL3'] +
                         ((hdr_gssm['NAXIS3'] - hdr_gssm['CRPIX3'])
                          * hdr_gssm['CDELT3']),
                         hdr_gssm['NAXIS3'])
# ############################################################################ #
# ############################################################################ #
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

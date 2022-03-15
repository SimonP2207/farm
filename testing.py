from pathlib import Path

import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
from astropy.coordinates import SkyCoord

import farm

nx, ny, cell_size = 512, 512, 8. / 512
ra, dec = "01:02:03.4", "05:06:07.89"

dims = (nx, ny)
coord = SkyCoord(ra, dec, frame='fk5', unit=(u.hourangle, u.degree))

gssm = farm.GSSM(dims, cell_size, coord, model='MHD')
gdsm = farm.GDSM(dims, cell_size, coord, model='GSM2016')
skymodel = farm.SkyModel(dims, cell_size, coord, gssm.frequencies)

gdsm.add_frequency(gssm.frequencies)
gssm.normalise(gdsm, inplace=True)

skymodel += (gssm, gdsm)

gdsm_cdelt = 0.0621480569243431  # Apparent pixel size of GDSM model
nx, ny, cell_size = 512, 512, 8. / 512
ra, dec = "01:02:03.4", "05:06:07.89"

farm.plotting_functions.plot_spix(skymodel)

gssm.write_fits(Path("test_gssm.fits"), unit='K')
gdsm.write_fits(Path("test_gdsm.fits"), unit='K')
skymodel.write_fits(Path("test_skymodel.fits"), unit='K')

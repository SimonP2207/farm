from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
import farm

dims  = 512, 512  # n_pixels in x/R.A., n_pixels in y/declination
field_of_view = 8.  # deg
cell_size = field_of_view / dims[0]  # deg

coord = SkyCoord("01:02:03.4", "05:06:07.89",
                 frame='fk5', unit=(u.hourangle, u.degree))

# Use MHD model for Galactic small-scale structure SkyComponent. This is loaded
# from data/Gsynch_SKAs.fits directly and thus its frequency information is
# fixed.
gssm = farm.SkyComponent.load_from_fits(fitsfile=farm.DATA_FILES['MHD'],
                                        name='GSSM', cdelt=cell_size,
                                        coord0=coord)

# Use GSM2016 model (Zheng et al, 2016) for Galactic diffuse-scale structure
# SkyComponent. Add the same frequencies as the GSSM sky component to enable
# combination.
gdsm = farm.SkyComponent(name='GDSM', npix=dims, cdelt=cell_size, coord0=coord,
                         tb_func=farm.tb_functions.gdsm2016_t_b)
gdsm.add_frequency(gssm.frequencies)

# Create SkyModel instance which will handle the combination of the various
# SkyComponent instances. Add frequency information to it
skymodel = farm.SkyModel(dims, cell_size, coord, gssm.frequencies)

# Add individual SkyComponent instances to the SkyModel
skymodel += (gssm, gdsm)  # Equivalent to skymodel.add_component((gssm, gdsm))

# Write the sky model to a .fits file
skymodel.write_fits(Path("test_skymodel_Inu.fits"), unit='JY/SR')

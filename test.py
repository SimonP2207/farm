import pathlib
import numpy as np
from astropy.io import fits
import farm



fits_trecs = pathlib.Path(
    '/Users/simon.purser/Dropbox (SKAO)/SDC3/Data/SkyModels/PS/trecs_sdc3_v1c20.fits'
)

fits_trecs_jyppix = pathlib.Path(str(fits_trecs).replace(
    '.fits', '_jyppix.fits')
)

fits_trecs = pathlib.Path(
    '/Users/simon.purser/Dropbox (SKAO)/SDC3/Data/SkyModels/PS') / \
    'sky_continuum_sdc3_v1_1.fits'


# hdulist = fits.open(fits_file)
# hdulist[0].header['BUNIT'] = 'JY/PIXEL'
#
# hdr = hdulist[0].header
#
# beam_area = farm.physics.astronomy.gaussian_beam_area(hdr['BMAJ'], hdr['BMIN'])
# cell_area = np.radians(np.abs(hdr['CDELT1'])) * np.radians(hdr['CDELT2'])
#
# hdulist[0].data *= cell_area / beam_area
# hdulist.writeto(fits_trecs_jyppix)

a = farm.sky_model.classes.SkyComponent.load_from_fits(
    fits_trecs_jyppix, freqs=np.array([1e8, 1.025e8, 1.05e8, 1.075e8, 1.1e8])
)

from astropy.io import fits
from astropy.io.fits import Header

from fits_test_data import *


def create_hdr(data: np.ndarray, cdelt: float, freqs: np.ndarray,
               coord0: SkyCoord, bunit: str = 'K'):
    n_freq, n_y, n_x = np.shape(data)
    hdr_dict = {
        'BITPIX': -32,
        'NAXIS': 2,
        'NAXIS1': 2,
        'NAXIS2': 2,
        'NAXIS3': len(freqs),
        'CTYPE1': 'RA---SIN',
        'CTYPE2': 'DEC--SIN',
        'CTYPE3': 'FREQ    ',
        'CRVAL1': coord0.ra.deg,
        'CRVAL2': coord0.dec.deg,
        'CRVAL3': min(freqs),
        'CRPIX1': n_x / 2,
        'CRPIX2': n_y / 2,
        'CRPIX3': 1,
        'CDELT1': -cdelt,
        'CDELT2': cdelt,
        'CDELT3': freqs[1] - freqs[0],
        'CUNIT1': 'deg     ',
        'CUNIT2': 'deg     ',
        'CUNIT3': 'Hz      ',
        'EQUINOX': 2000.,
        'BUNIT': format(bunit, '8'),
    }

    # Guarantee order in which keywords are added to .fits header
    order = ('BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
             'CTYPE1', 'CTYPE2', 'CTYPE3', 'CRVAL1', 'CRVAL2', 'CRVAL3',
             'CRPIX1', 'CRPIX2', 'CRPIX3', 'CDELT1', 'CDELT2', 'CDELT3',
             'CUNIT1', 'CUNIT2', 'CUNIT3', 'EQUINOX', 'BUNIT')

    hdr = Header({'Simple': True})
    for keyword, value in ((kw, hdr_dict[kw]) for kw in order):
        hdr.set(keyword, value)

    return hdr


def create_fits(data: np.ndarray, header: Header, fitsfile: str):
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdu.writeto(fitsfile, overwrite=True)


# Headers
hdr1_tb = create_hdr(test_tb1, test_cdelt, test_frequencies, test_coord0,
                     bunit='K')
hdr2_tb = create_hdr(test_tb2, test_cdelt, test_frequencies, test_coord0,
                     bunit='K')
hdr1_inu = create_hdr(test_inu1, test_cdelt, test_frequencies, test_coord0,
                      bunit='JY/SR')
hdr2_inu = create_hdr(test_inu2, test_cdelt, test_frequencies, test_coord0,
                      bunit='JY/SR')
hdr1_snu = create_hdr(test_snu1, test_cdelt, test_frequencies, test_coord0,
                      bunit='JY/PIXEL')
hdr2_snu = create_hdr(test_snu2, test_cdelt, test_frequencies, test_coord0,
                      bunit='JY/PIXEL')

# Create .fits cubes
fitsfile1 = 'test_fits1_{}.fits'
fitsfile2 = 'test_fits2_{}.fits'
create_fits(test_tb1, hdr1_tb, fitsfile1.format('tb'))
create_fits(test_tb2, hdr2_tb, fitsfile2.format('tb'))
create_fits(test_inu1, hdr1_inu, fitsfile1.format('inu'))
create_fits(test_inu2, hdr2_inu, fitsfile2.format('inu'))
create_fits(test_snu1, hdr1_snu, fitsfile1.format('snu'))
create_fits(test_snu2, hdr2_snu, fitsfile2.format('snu'))

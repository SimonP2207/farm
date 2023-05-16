"""
All methods/classes related to image handling/manipulation
"""
import shutil
import pathlib
from typing import Union, Tuple, Optional
import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS

from .. import LOGGER


def pb_multiply(in_image: pathlib.Path, pb: pathlib.Path,
                out_fitsfile: pathlib.Path, cellscal: str = 'CONSTANT'):
    """
    Multiply (not divide i.e. 'pbcor') an image by the primary beam

    Parameters
    ----------
    in_image
        Image to multiply by the primary beam
    pb
        Primary beam to multiply image with
    out_fitsfile
        Full path to output image containing image multiplied by the primary
        beam
    cellscal
        Whether to scale cell size with the inverse of frequency ('1/F') or not
        ('CONSTANT', the default). Must be one of '1/F' or 'CONSTANT'
    """
    from . import error_handling as errh
    from . import generate_random_chars as grc
    from ..software.miriad import miriad

    if cellscal.lower() not in ('constant', '1/f'):
        errh.raise_error(ValueError,
                         "cellscal must be one of 'CONSTANT' or '1/F', "
                         f"not {cellscal}")
    cellscal = cellscal.upper()

    LOGGER.info(f"Multiplying {in_image} by beam response, {pb} with "
                f"{cellscal} cell size scaling in frequency")

    # Convert input image and PB to miriad image format if necessary
    if not is_miriad_image(pb):
        pb_mirim = pb.with_suffix('.im')
        miriad.fits(op="xyin", _in=pb, out=pb_mirim)
    else:
        pb_mirim = pb

    if not is_miriad_image(in_image):
        in_image_mirim = in_image.with_suffix('.im')
        miriad.fits(op='xyin', _in=in_image, out=in_image_mirim)
    else:
        in_image_mirim = in_image

    expr = f"<{in_image_mirim}>*<{pb_mirim}>"
    out_mirim = out_fitsfile.with_suffix('.im')
    if cellscal == '1/F':
        naxis3 = int(miriad.gethd(_in=f'{pb_mirim}/naxis3'))
        crpix3 = int(miriad.gethd(_in=f'{pb_mirim}/crpix3'))
        cdelt3 = float(miriad.gethd(_in=f'{pb_mirim}/cdelt3'))  # GHz
        crval3 = float(miriad.gethd(_in=f'{pb_mirim}/crval3'))  # GHz
        freq_min = crval3 - cdelt3 * (crpix3 - 1)
        freq_max = crval3 + cdelt3 * (naxis3 - crpix3)

        expr += "/z**2"
        zs = f"1,{freq_max / freq_min:.6f}"
        temp_out_mirim = out_mirim.parent / f'temp_{grc(10)}.mirim'
        temp2_out_mirim = out_mirim.parent / f'temp_{grc(10)}.mirim'
        try:
            miriad.maths(exp=expr, out=temp_out_mirim, zrange=zs)
            shutil.copytree(temp_out_mirim, temp2_out_mirim)
            miriad.puthd(_in=f"{temp2_out_mirim}/cellscal", value=cellscal)
            miriad.regrid(_in=temp_out_mirim, tin=temp2_out_mirim,
                          out=out_mirim)
        finally:
            if temp_out_mirim.exists():
                shutil.rmtree(temp_out_mirim)
            if temp2_out_mirim.exists():
                shutil.rmtree(temp2_out_mirim)
    else:
        miriad.maths(exp=expr, out=out_mirim)

    miriad.fits(_in=out_mirim, out=out_fitsfile, op='xyout')


def regrid_fits(fits_in: pathlib.Path, template_im: pathlib.Path,
                fits_out: Optional[pathlib.Path] = None, inplace=True):
    """
    Regrid a .fits file using a template (either a miriad image or .fits)

    Parameters
    ----------
    fits_in
        Full path to .fits image to be regridded
    template_im
        Full path to miriad image, or .fits image, providing the coordinate
        system to regrid to
    fits_out
        Full path to write regridded .fits image to. If None, inplace must be
        True. Default is None
    inplace
        Whether to regrid the input .fits image in place or write a new .fits
        image to fits_out. If inplace is False, fits_out must not be None.
        Default is True
    """
    from . import generate_random_chars as grc
    from ..software.miriad import miriad

    LOGGER.info(f"Regridding {fits_in} using {template_im} as a template")
    temp_mir_im_in = f'temp_mirim_in_{grc(10)}.im'
    temp_mir_im_out = f'temp_mirim_{grc(10)}.im'
    miriad.fits(_in=fits_in, op='xyin', out=temp_mir_im_in)

    if not is_miriad_image(template_im):
        LOGGER.info(f"Converting template, {template_im}, from fits to miriad "
                    f"image for regridding")
        temp_mir_im_template = f'temp_mirim_template_{grc(10)}.im'
        miriad.fits(_in=template_im, op='xyin', out=temp_mir_im_template)
        miriad.regrid(_in=temp_mir_im_in, tin=temp_mir_im_template,
                      out=temp_mir_im_out)
        shutil.rmtree(temp_mir_im_template)

    else:
        miriad.regrid(_in=temp_mir_im_in, tin=template_im, out=temp_mir_im_out)

    if inplace:
        fits_in.unlink()
        fits_out = fits_in

    LOGGER.info(f"Writing regridded image to {fits_out}")
    miriad.fits(_in=temp_mir_im_out, op='xyout', out=fits_out)

    shutil.rmtree(temp_mir_im_in)
    shutil.rmtree(temp_mir_im_out)


def is_miriad_image(mir_im: pathlib.Path):
    """Check if an image is a miriad image"""
    if not mir_im.exists():
        return False
    if not mir_im.is_dir():
        return False

    contents = [_.name for _ in mir_im.iterdir()]
    req_contents = ('header', 'history', 'image')

    return all([req_content in contents for req_content in req_contents])


def calculate_spix(data_cube: npt.ArrayLike,
                   frequencies: npt.ArrayLike,
                   interpolate_nans: bool = True
                   ) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculate the spectral index across a SubbandSkyModel or Skycomponent
    instance field of view, using standard least squares regression across all
    provided frequencies

    Parameters
    ----------
    data_cube
        Data cube of three dimensions (FREQ, X, Y)
    frequencies
        Numpy array of frequencies corresponding to 0-th axis of data-cube
    interpolate_nans
        Whether to interpolate spectral indices between neighbouring channels in
        the case the least squares regression calculation fails due to negative
        data. Default is True

    Returns
    -------
    Spectral indices as a 2-dimensional np.ndarray of shape
    (sky_model_type.n_y, sky_model_type.n_x) and 'y-intercept' of least squares
    fit
    """
    # spix = (np.log10(data_cube[0] / data_cube[-1]) /
    #         np.log10(frequencies[0] / frequencies[-1]))

    # LSQ regression line
    log_data_cube = np.log10(data_cube)
    log_frequencies = np.reshape(np.log10(frequencies),
                                 (len(frequencies), 1, 1))

    n = len(frequencies)
    sum_xy = np.sum(log_frequencies * log_data_cube, axis=0)
    sum_y = np.sum(log_data_cube, axis=0)
    sum_x = np.sum(log_frequencies)
    sum_x2 = np.sum(log_frequencies ** 2., axis=0)

    c = ((sum_y * sum_x2) - sum_x * sum_xy) / (n * sum_x2 - sum_x ** 2.)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2.)

    if interpolate_nans:
        # First interpolate between beginning/end channels
        spix = (np.log10(data_cube[0] / data_cube[-1]) /
                np.log10(frequencies[0] / frequencies[-1]))
        m = np.where(np.isnan(m), spix, m)

        for i in range(len(frequencies) - 1):
            spix = (np.log10(data_cube[0] / data_cube[i + 1]) /
                    np.log10(frequencies[0] / frequencies[i + 1]))
            m = np.where(np.isnan(m), spix, m)

    return m, c


def crop_frac(img: np.ndarray, fracx: float, fracy: float) -> np.ndarray:
    z, y, x = img.shape
    cropx = int(x * fracx)
    cropy = int(y * fracy)
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]


def unwrap_angle(angle: float) -> float:
    """Unwraps angles > 2 pi radians or angles < -2 pi radians"""
    return angle % (np.sign(angle) * 2. * np.pi)


def rotate_image(image_data: np.ndarray, phi: float,
                 x_axis: int = 2, y_axis: int = 1) -> np.ndarray:
    """
    Rotate an image by a defined angle. First rotates by integer multiples of 90
    to bring remaining rotation in the range -45deg <= phi <= 45 deg, then uses
    the FFT-based shear rotation of Larkin et al. (1997) to perform the final
    rotation range

    Parameters
    ----------
    image_data
        Array of image data
    phi
        Angle with which to rotate image clockwise [radians]
    x_axis
        Axis within numpy array representing image x-axis (i.e. right ascension)
    y_axis
        Axis within numpy array representing image y-axis (i.e. declination)

    Returns
    -------
    np.ndarray of rotated image data
    """
    image_data = np.squeeze(image_data)
    axes = (x_axis, y_axis)

    phip = unwrap_angle(phi)

    # Calculate number of, and perform, basic, 90-degree image rotations
    n90rots, fac = {
        -9 * np.pi / 4. <= phip < -7 * np.pi / 4.: (-0, +4. * np.pi / 2.),
        -7 * np.pi / 4. <= phip < -5 * np.pi / 4.: (-3, +3. * np.pi / 2.),
        -5 * np.pi / 4. <= phip < -3 * np.pi / 4.: (-2, +2. * np.pi / 2.),
        -3 * np.pi / 4. <= phip < -1 * np.pi / 4.: (-1, +1. * np.pi / 2.),
        -1 * np.pi / 4. <= phip < +1 * np.pi / 4.: (+0, +0. * np.pi / 2.),
        +1 * np.pi / 4. <= phip < +3 * np.pi / 4.: (+1, -1. * np.pi / 2.),
        +3 * np.pi / 4. <= phip < +5 * np.pi / 4.: (+2, -2. * np.pi / 2.),
        +5 * np.pi / 4. <= phip < +7 * np.pi / 4.: (+3, -3. * np.pi / 2.),
        +7 * np.pi / 4. <= phip < +9 * np.pi / 4.: (+0, -4. * np.pi / 2.)
    }[True]

    image_data = np.rot90(image_data, k=n90rots, axes=axes)  # axes was (1, 2)
    phip += fac  # Remaining angle of rotation for FFT-based shear rotation

    # Calculate border padding on each axis
    bord0 = int(image_data.shape[1] * 4 / 16)
    bord1 = int(image_data.shape[1] * 4 / 16)

    # pad first with reflected input image
    impad0 = np.pad(image_data, ((0, 0), (bord0, bord0), (bord0, bord0)),
                    mode='reflect')
    # pad some more with zeros
    impad = np.pad(impad0, ((0, 0), (bord1, bord1), (bord1, bord1)))
    zlen, ylen, xlen = impad.shape
    ufreq = np.fft.fftfreq(xlen)
    gx = np.zeros((zlen, ylen, xlen))
    gyx = np.zeros((zlen, ylen, xlen))
    gxyx = np.zeros((zlen, ylen, xlen))

    a, b = np.tan(phip / 2.), -np.sin(phip)
    # do FFT-shear-based rotation following Larkin et al. 1997 (only for
    # rotations of -45 deg <= phi <= +45 deg)
    for iplan in range(0, zlen, 1):
        for icol in range(0, xlen, 1):
            gx[iplan, icol, :] = np.real(np.fft.ifft(
                np.fft.fft(impad[iplan, icol, :]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * a)
            ))

        for icol in range(0, xlen, 1):
            gyx[iplan, :, icol] = np.real(np.fft.ifft(
                np.fft.fft(gx[iplan, :, icol]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * b)
            ))

        for icol in range(0, xlen, 1):
            gxyx[iplan, icol, :] = np.real(np.fft.ifft(
                np.fft.fft(gyx[iplan, icol, :]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * a)
            ))

    return crop_frac(gxyx, 0.5, 0.5)


def gaussian_2d(x: Union[float, npt.ArrayLike], y: Union[float, npt.ArrayLike],
                x0: float, y0: float, peak: float, major: float, minor: float,
                pa: float) -> Union[float, npt.ArrayLike]:
    """


    Parameters
    ----------
    x
        Right ascension-coordinate(s) [deg]
    y
        Declination-coordinate(s) [deg]
    x0
        Peak right ascension-coordinate [deg]
    y0
        Peak declination-coordinate [deg]
    peak
        Peak intensity value [Jy/pixel]
    major
        Major axis FWHM [arcsec]
    minor
        Minor axis FWHM [arcsec]
    pa
        Major axis position angle (east from north) [deg]

    Returns
    -------
    Intensity(s) [Jy/pixel]
    """
    pa_rad = np.radians(pa)
    sigma_maj2 = (major / 3600. / 2.35482) ** 2.
    sigma_min2 = (minor / 3600. / 2.35482) ** 2.
    cos_pa_2 = np.cos(pa_rad) ** 2.
    sin_pa_2 = np.sin(pa_rad) ** 2.
    sin_2pa = np.sin(2. * pa_rad)

    a = cos_pa_2 / (2. * sigma_min2) + sin_pa_2 / (2. * sigma_maj2)
    b = -sin_2pa / (4. * sigma_min2) + sin_2pa / (4. * sigma_maj2)
    c = sin_pa_2 / (2. * sigma_min2) + cos_pa_2 / (2. * sigma_maj2)

    dx, dy = x - x0, y - y0
    return peak * np.exp(-(a * dx ** 2. + 2. * dx * dy * b + c * dy ** 2.))


def place_point_source_on_grid(data: npt.NDArray, im_wcs: WCS, tgt_ra: float,
                               tgt_dec: float, tgt_flux0: float,
                               tgt_freq0: float, tgt_spix: float):
    """
    Given a coordinate for a point source, add it (in place) to the data array
    of fluxes

    Parameters
    ----------
    data
        Data array of fluxes
    im_wcs
        World coordinate system
    tgt_ra
        Source's right ascension [deg]
    tgt_dec
        Source's declination [deg]
    tgt_flux0
        Source's peak-flux/flux at freq0 [Jy]
    tgt_freq0
        Frequency of source's peak flux [Hz]
    tgt_spix
        Spectral index of source
    """
    im_hdr = im_wcs.to_header(relax=True)
    nfreq = im_wcs.spectral.array_shape[0]
    im_freqs = np.array(
        [im_hdr['CRVAL3'] + (n - im_hdr['CRPIX3'] + 1) * im_hdr['CDELT3']
         for n in range(nfreq)]
    )

    # Get size of pixel in both RA and declination
    d_dec = im_hdr['CDELT2']
    d_ra = d_dec / np.cos(np.radians(tgt_dec))
    cell_area = np.abs(d_ra * d_dec)

    verts = np.array(
        [[[tgt_ra + d_ra / 2., tgt_dec - d_dec / 2.],
          [tgt_ra - d_ra / 2., tgt_dec - d_dec / 2.],
          [tgt_ra + d_ra / 2., tgt_dec + d_dec / 2.],
          [tgt_ra - d_ra / 2., tgt_dec + d_dec / 2.]]] * nfreq
    )
    ras = verts[:, :, 0].reshape((nfreq, 2, 2))
    decs = verts[:, :, 1].reshape((nfreq, 2, 2))
    freqs = np.concatenate([np.full((1, 2, 2), freq) for freq in im_freqs],
                           axis=0)

    crds = np.concatenate((ras[..., np.newaxis], decs[..., np.newaxis]), axis=3)
    idxs_verts = im_wcs.world_to_array_index_values(ras, decs, freqs)
    cell_crds = im_wcs.array_index_to_world_values(*idxs_verts)[:2]

    # If source is near RA = 0, imfunc.gaussian_2d is given coordinates
    # in ra_ that it calculates are ~360deg from source position (due
    # to the wrapped nature of RA, which imfunc.gaussian_2d is unaware
    # of) leading to zeroes. Therefore unwrap the ra_ coordinates if
    # required
    if np.ptp(cell_crds[0]) > 180:
        cell_crds = (np.where(np.abs(cell_crds[0] - tgt_ra) > 180,
                     cell_crds[0] + (360. if tgt_ra > 180 else -360.),
                     cell_crds[0]),
                     cell_crds[1])

    crd_intersection = np.mean(cell_crds, axis=(1, 2, 3))

    outside_arr = ((idxs_verts[2] >= data.shape[2]) | (idxs_verts[2] < 0) |
                   (idxs_verts[1] >= data.shape[1]) | (idxs_verts[1] < 0))

    # Do not add source if any part of it lies outside data array
    if True in outside_arr:
        return verts, crd_intersection

    offsets = crds - crd_intersection
    areas = np.abs(np.prod(offsets, axis=(3,))) / cell_area

    vals = (
        areas[np.newaxis, ...] * tgt_flux0 *
        (im_freqs[..., np.newaxis, np.newaxis] / tgt_freq0) ** tgt_spix
    )

    data[idxs_verts] = vals


def place_gaussian_on_grid(data: npt.NDArray, ras: npt.NDArray,
                           decs: npt.NDArray, freqs: npt.NDArray,
                           im_wcs: WCS, tgt_ra: float,
                           tgt_dec: float, tgt_flux0: float,
                           tgt_freq0: float, tgt_spix: float,
                           tgt_maj_as: float, tgt_min_as: float,
                           tgt_pa_deg: float):
    """
    Given a coordinate and dimensions for a Gaussian source, add it (in place)
    to the data array of fluxes

    Parameters
    ----------
    data
        Data array of fluxes [Jy/pixel]
    ras
        Grid of right ascension coordinates of the same shape as data [deg]. See
        make_coordinate_grid for an acceptable format
    decs
        Grid of declination coordinates of the same shape as data [deg]. See
        make_coordinate_grid for an acceptable format
    freqs
        Grid of frequency coordinates of the same shape as data [Hz]. See
        make_coordinate_grid for an acceptable format
    im_wcs
        World coordinate system
    tgt_ra
        Source's right ascension [deg]
    tgt_dec
        Source's declination [deg]
    tgt_flux0
        Source's peak-flux/flux at freq0 [Jy]
    tgt_freq0
        Frequency of source's peak flux [Hz]
    tgt_spix
        Spectral index of source
    tgt_maj_as
        Source's major axis FWHM [arcsec]
    tgt_min_as
        Source's minor axis FWHM [arcsec]
    tgt_pa_deg
        Source's major axis' position angle (east from north) [deg]
    """
    cdelt1, cdelt2, cdelt3 = im_wcs.wcs.cdelt
    naxis1, naxis2, naxis3 = im_wcs.pixel_shape
    crpix1, crpix2, crpix3 = im_wcs.wcs.crpix
    crval1, crval2, crval3 = im_wcs.wcs.crval

    im_freqs = np.array(
        [crval3 + (n - crpix3 + 1) * cdelt3 for n in range(naxis3)]
    )

    didx = np.max([int(tgt_maj_as * 2 / 3600. // cdelt2 + 1), 5])
    idxs_target = im_wcs.world_to_array_index_values(
        tgt_ra, tgt_dec, im_freqs[0]
    )
    ra_idx = (idxs_target[2] - didx, idxs_target[2] + didx + 1)
    dec_idx = (idxs_target[1] - didx, idxs_target[1] + didx + 1)

    # Index ranges in ra and dec within which to calculate source
    # flux
    ra_idx = (np.max([ra_idx[0], 0]), np.min([ra_idx[1], naxis1]))
    dec_idx = (np.max([dec_idx[0], 0]), np.min([dec_idx[1], naxis2]))

    ra_ = ras[:, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]]
    dec_ = decs[:, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]]
    freqs_ = freqs[:, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]]

    # If source is near RA = 0, imfunc.gaussian_2d is given coordinates
    # in ra_ that it calculates are ~360deg from source position (due
    # to the wrapped nature of RA, which imfunc.gaussian_2d is unaware
    # of) leading to zeroes. Therefore unwrap the ra_ coordinates if
    # required
    if np.ptp(ra_) > 180:
        ra_ = np.where(np.abs(ra_ - tgt_ra) > 180,
                       ra_ + (360. if tgt_ra > 180 else -360.),
                       ra_)

    # For sources with Gaussian shapes
    val0 = gaussian_2d(
        ra_, dec_, tgt_ra, tgt_dec, 1,
        np.max([tgt_maj_as, np.abs(cdelt2) * 3600]),
        np.max([tgt_min_as, np.abs(cdelt2) * 3600]), tgt_pa_deg
    )

    # Normalise
    val0 *= tgt_flux0 / np.nansum(val0, axis=(1, 2))[..., np.newaxis, np.newaxis]

    vals = val0 * (freqs_/ tgt_freq0) ** tgt_spix
    data[:, dec_idx[0]:dec_idx[1], ra_idx[0]:ra_idx[1]] += vals


def make_coordinate_grid(im_wcs: WCS) -> npt.NDArray:
    """
    Create and return (as a 3-tuple) grids of ra, dec and frequency given a
    world coordinate system
    """
    naxis1, naxis2, naxis3 = im_wcs.pixel_shape
    zz, yy, xx = np.meshgrid(np.arange(naxis3),
                             np.arange(naxis2),
                             np.arange(naxis1), indexing='ij')
    return im_wcs.wcs_pix2world(xx, yy, zz, 0)


def deconvolve(image, psf):
    from scipy import fftpack

    star_fft = fftpack.fftshift(fftpack.fftn(image))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))

    return fftpack.fftshift(
        fftpack.ifftn(fftpack.ifftshift(star_fft / psf_fft)))

"""
All methods/classes related to image handling
"""
from typing import Union
import numpy as np
import numpy.typing as npt
from ..sky_model import SkyClassType


def calculate_spix(sky_model_type: SkyClassType) -> np.ndarray:
    """
    Calculate the spectral index across a SkyModel or Skycomponent instance
    field of view

    Parameters
    ----------
    sky_model_type
        SkyModel or SkyComponent

    Returns
    -------
    Spectral indices as a 2-dimensional np.ndarray of shape
    (sky_model_type.n_y, sky_model_type.n_x)
    """
    spix = (np.log10(sky_model_type.data('JY/PIXEL')[0] /
                     sky_model_type.data('JY/PIXEL')[-1]) /
            np.log10(sky_model_type.frequencies[0] /
                     sky_model_type.frequencies[-1]))
    return spix


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
            gx[iplan, icol, :] = np.real(np.fft.ifft(np.fft.fft(
                impad[iplan, icol, :]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * a)))

        for icol in range(0, xlen, 1):
            gyx[iplan, :, icol] = np.real(np.fft.ifft(np.fft.fft(
                gx[iplan, :, icol]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * b)))

        for icol in range(0, xlen, 1):
            gxyx[iplan, icol, :] = np.real(np.fft.ifft(np.fft.fft(
                gyx[iplan, icol, :]) *
                np.exp(-2.j * np.pi * (icol - xlen / 2) * ufreq * a)))

    return crop_frac(gxyx, 0.5, 0.5)


def gaussian_2d(x: Union[float, npt.ArrayLike], y: Union[float, npt.ArrayLike],
                x0: float, y0: float, peak: float, major: float, minor: float,
                pa: float) -> Union[float, npt.ArrayLike]:
    """


    Parameters
    ----------
    x
        x/right ascension-coordinate(s) [deg]
    y
        y/declination-coordinate(s) [deg]
    x0
        Peak x/right ascension-coordinate [deg]
    y0
        Peak y/declination-coordinate [deg]
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


def deconvolve(image, psf):
    from scipy import fftpack

    star_fft = fftpack.fftshift(fftpack.fftn(image))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))

    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))

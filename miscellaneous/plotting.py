"""
Contains all plotting related functionality, matplotlib being data visualisation
tool-of-choice
"""
import pathlib
from typing import Union, Tuple, Optional, List
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.pylab as plt
import matplotlib.axes
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..sky_model.classes import SkyClassType
from ..data.loader import Telescope
from .image_functions import calculate_spix
from . import decorators

SUBPLOT_WIDTH_INCHES = 3.32153
COLORBAR_WIDTH_FRAC = 0.1  # Colorbar width as percentage of parent axes

assert 0 < COLORBAR_WIDTH_FRAC < 1, "Invalid fractional colorbar width"


def flux(sky_model_type: Union[None, SkyClassType] = None,
         data_cube: Union[None, npt.ArrayLike] = None,
         frequencies: Union[None, npt.ArrayLike] = None,
         extent: Union[None, Tuple[float, float]] = None,
         precision: int = 3,
         vmin: Union[None, float] = None,
         vmax: Union[None, float] = None,
         ax: Union[None, matplotlib.axes.Axes] = None,
         cax: Union[None, matplotlib.axes.Axes] = None,
         savefig: Union[str, bool] = False
         ) -> Tuple[matplotlib.figure.Figure,
                    Tuple[matplotlib.axes.Axes,
                          matplotlib.axes.Axes]]:
    """
    Plot of the spectral index across a SkyModel of SkyComponent instance

    Parameters
    ----------
    sky_model_type
        SkyModel or SkyComponent instance
    data_cube
        3D np.ndarray (FREQ, X, Y) containing intensities/fluxes. If None, it is
         parsed from sky_model_type
    frequencies
         1D np.ndarray containing frequencies of channels corresponding to
         zeroth axis of data_cube. If None, it is parsed from sky_model_type
    extent
        Tuple containing extent of x and y-axes [deg]. If None, it is parsed
        from sky_model_type. If sky_model_type is None, values of (1., 1.) are
        assigned
    precision
        Number of decimal points of precision on spectral index scale
    vmin
        Minimum flux on colourscale
    vmax
        Maximum flux on colourscale
    ax
        matplotlib.axes.Axes instance to plot onto. If None (default),
        matplotlib.figure.Figure and matplotlib.axes.Axes instances are created
    cax
        matplotlib.axes.Axes instance to use for colorbar. If None, one is
        created by taking space from ax
    savefig
        Full path to save the resulting figure to. If False, figure is not saved
        (default)

    Returns
    -------
    Tuple of matplotlib.figure.Figure, (matplotlib.axes.Axes,
    matplotlib.axes.Axes) which corresponding to (figure, (plot axis, colorbar
    axis)

    Raises
    ------
    ValueError
        If sky_model_type is not specified and one of data_cube or frequencies
        is not specified either
    """
    if sky_model_type is not None:
        data_cube = sky_model_type.data("JY/PIXEL")
        extent = (sky_model_type.n_x * sky_model_type.cdelt,
                  sky_model_type.n_y * sky_model_type.cdelt)
    else:
        if None in (data_cube, frequencies):
            raise ValueError("If sky_model_type is not specified, data_cube "
                             "and frequencies must not be None")

    # data_cube += 0.0033611419381985564 - np.nanmean(data_cube)
    flux_ = np.nanmean(data_cube, axis=0)

    fac_precision = 1.0
    for i in range(precision):
        fac_precision /= 10.0

    if vmin is None:
        vmin = np.floor(np.nanmin(flux_) / fac_precision) * fac_precision
    if vmax is None:
        vmax = np.ceil(np.nanmax(flux_) / fac_precision) * fac_precision

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1,
                               figsize=(SUBPLOT_WIDTH_INCHES /
                                        (1 + COLORBAR_WIDTH_FRAC),
                                        SUBPLOT_WIDTH_INCHES))
    else:
        fig = ax.figure

    if cax is None:
        from mpl_toolkits.axes_grid1.axes_size import Fraction, AxesX

        divider = make_axes_locatable(ax)
        cbar_size = Fraction(COLORBAR_WIDTH_FRAC / (1.0 - COLORBAR_WIDTH_FRAC),
                             AxesX(ax))
        cax = divider.append_axes("right", size=cbar_size, pad=0.0)

    cmap = cm.get_cmap('plasma', 12)

    if extent is not None:
        im = ax.imshow(flux_, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=(+extent[0] / 2, -extent[0] / 2,
                               -extent[1] / 2, +extent[1] / 2))
    else:
        ny, nx = np.shape(data_cube)[1:]
        im = ax.imshow(flux_, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=(+nx / 2, -nx / 2, -ny / 2, +ny / 2))

    plt.colorbar(im, cax=cax, ax=ax, orientation='vertical')
    cax.xaxis.tick_top()

    ax.set_xticks(range(-4, 5)[::-1])
    ax.set_yticks(range(-4, 5))

    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.tick_params(which='both', axis='both', bottom=True, top=True, left=True,
                   right=True, direction='in')

    if extent:
        ax.set_xlabel(r'$\Delta \, \mathrm{R.A.\, \left[deg\right]}$')
        ax.set_ylabel(r'$\Delta \, \mathrm{Dec.\, \left[deg\right]}$')
    else:
        ax.set_xlabel(r'$\Delta \, \mathrm{R.A.\, \left[pixel\right]}$')
        ax.set_ylabel(r'$\Delta \, \mathrm{Dec.\, \left[pixel\right]}$')

    # cax.set_yticks(np.arange(vmin, vmax + fac_precision, fac_precision))

    cax.text(0.5, 0.5, r'$S_\nu \, \left[ \mathrm{Jy \, pixel^{-1}} \right]$',
             transform=cax.transAxes, rotation=90., color='w',
             horizontalalignment='center', verticalalignment='center')

    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, (ax, cax)


def spix(sky_model_type: Union[None, SkyClassType] = None,
         data_cube: Union[None, npt.ArrayLike] = None,
         frequencies: Union[None, npt.ArrayLike] = None,
         extent: Union[None, Tuple[float, float]] = None,
         precision: int = 1,
         vmin: Union[None, float] = None,
         vmax: Union[None, float] = None,
         ax: Union[None, matplotlib.axes.Axes] = None,
         cax: Union[None, matplotlib.axes.Axes] = None,
         savefig: Union[str, bool] = False
         ) -> Tuple[matplotlib.figure.Figure,
                    Tuple[matplotlib.axes.Axes,
                          matplotlib.axes.Axes]]:
    """
    Plot of the spectral index across a SkyModel of SkyComponent instance

    Parameters
    ----------
    sky_model_type
        SkyModel or SkyComponent instance
    data_cube
        3D np.ndarray (FREQ, X, Y) containing intensities/fluxes. If None, it is
         parsed from sky_model_type
    frequencies
         1D np.ndarray containing frequencies of channels corresponding to
         zeroth axis of data_cube. If None, it is parsed from sky_model_type
    extent
        Tuple containing extent of x and y-axes [deg]. If None, it is parsed
        from sky_model_type. If sky_model_type is None, values of (1., 1.) are
        assigned
    precision
        Number of decimal points of precision on spectral index scale
    vmin
        Minimum spectral index on colourscale
    vmax
        Maximum spectral index on colourscale
    ax
        matplotlib.axes.Axes instance to plot onto. If None (default),
        matplotlib.figure.Figure and matplotlib.axes.Axes instances are created
    cax
        matplotlib.axes.Axes instance to use for colorbar. If None, one is
        created by taking space from ax
    savefig
        Full path to save the resulting figure to. If False, figure is not saved
        (default)

    Returns
    -------
    Tuple of matplotlib.figure.Figure, (matplotlib.axes.Axes,
    matplotlib.axes.Axes) which corresponding to (figure, (plot axis, colorbar
    axis)

    Raises
    ------
    ValueError
        If sky_model_type is not specified and one of data_cube or frequencies
        is not specified either
    """
    if sky_model_type is not None:
        data_cube = sky_model_type.data("JY/PIXEL")
        frequencies = sky_model_type.frequencies
        extent = (sky_model_type.n_x * sky_model_type.cdelt,
                  sky_model_type.n_y * sky_model_type.cdelt)
    else:
        if None in (data_cube, frequencies):
            raise ValueError("If sky_model_type is not specified, data_cube "
                             "and frequencies must not be None")

    # data_cube += 0.0033611419381985564 - np.nanmean(data_cube)
    spix_, _ = calculate_spix(data_cube, frequencies, interpolate_nans=False)

    fac_precision = 1.0
    for i in range(precision):
        fac_precision /= 10.0

    if vmin is None:
        vmin = np.floor(np.nanmin(spix_) / fac_precision) * fac_precision
    if vmax is None:
        vmax = np.ceil(np.nanmax(spix_) / fac_precision) * fac_precision

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(SUBPLOT_WIDTH_INCHES,
                                              SUBPLOT_WIDTH_INCHES * 1.05))
    else:
        fig = ax.figure

    if cax is None:
        from mpl_toolkits.axes_grid1.axes_size import Fraction, AxesX

        divider = make_axes_locatable(ax)
        cbar_size = Fraction(COLORBAR_WIDTH_FRAC / (1.0 - COLORBAR_WIDTH_FRAC),
                             AxesX(ax))
        cax = divider.append_axes("right", size=cbar_size, pad=0.0)

    cmap = cm.get_cmap('jet', 12)

    if extent is not None:
        im = ax.imshow(spix_, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=(+extent[0] / 2, -extent[0] / 2,
                               -extent[1] / 2, +extent[1] / 2))
    else:
        ny, nx = np.shape(data_cube)[1:]
        im = ax.imshow(spix_, vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=(+nx / 2, -nx / 2, -ny / 2, +ny / 2))

    plt.colorbar(im, cax=cax, ax=ax, orientation='vertical')
    cax.xaxis.tick_top()

    ax.set_xticks(range(-4, 5)[::-1])
    ax.set_yticks(range(-4, 5))

    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.tick_params(which='both', axis='both', bottom=True, top=True, left=True,
                   right=True, direction='in')

    if extent:
        ax.set_xlabel(r'$\Delta \, \mathrm{R.A.\, \left[deg\right]}$')
        ax.set_ylabel(r'$\Delta \, \mathrm{Dec.\, \left[deg\right]}$')
    else:
        ax.set_xlabel(r'$\Delta \, \mathrm{R.A.\, \left[pixel\right]}$')
        ax.set_ylabel(r'$\Delta \, \mathrm{Dec.\, \left[pixel\right]}$')

    cax.set_yticks(np.arange(vmin, vmax + fac_precision, fac_precision))

    cax.text(0.5, 0.5, r'$\alpha$', transform=cax.transAxes, color='w',
             horizontalalignment='center', verticalalignment='center',
             rotation=90.,)

    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, (ax, cax)


def power_spectrum(sky_model_type: SkyClassType, freq: float,
                   ax: Union[None, matplotlib.axes.Axes] = None,
                   savefig: Union[str, bool] = False):
    """

    Parameters
    ----------
    sky_model_type
        SkyModel or SkyComponent instance
    freq
        Frequency at which to plot the power-spectrum
    ax
        matplotlib.axes.Axes instance to plot onto. If None (default),
        matplotlib.figure.Figure and matplotlib.axes.Axes instances are created
    savefig
        Full path to save the resulting figure to. If False, figure is not saved
        (default)

    Returns
    -------
    Tuple of matplotlib.figure.Figure, matplotlib.axes.Axes, corresponding to
    (figure, plot axis)
    """
    import powerbox as pbox

    p_k_field, bins_field = pbox.get_power(sky_model_type.t_b(freq),
                                           (0.002, 0.002,))

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(SUBPLOT_WIDTH_INCHES,
                                              SUBPLOT_WIDTH_INCHES* 1.05))
    else:
        fig = ax.figure

    ax.plot(bins_field, p_k_field, 'b-')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if not savefig:
        plt.show()
    else:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, ax


# noinspection PyUnresolvedReferences
@decorators.suppress_warnings("astropy", "erfa")
def target_altaz(t0: Time,
                 tscop_location: EarthLocation,
                 coord_target: SkyCoord,
                 ax: Union[None, matplotlib.axes.Axes] = None,
                 cax: Union[None, matplotlib.axes.Axes] = None,
                 scan_times: Union[None, List[Tuple[Time, Time]]] = None,
                 savefig: Union[str, bool] = False
                 ) -> Tuple[matplotlib.figure.Figure,
                            Tuple[matplotlib.axes.Axes,
                                  matplotlib.axes.Axes]]:
    """
    Plot target altitude and azimuth as a function of time with colorscale of
    the elevation curve being the azimuth

    Parameters
    ----------
    t0
        Start-time of first scan
    tscop_location
        Location of the telescope
    coord_target
        Celestial coordinate of the pointing centre
    ax
        matplotlib.axes.Axes instance to plot onto. If None (default),
        matplotlib.figure.Figure and matplotlib.axes.Axes instances are created
    cax
        matplotlib.axes.Axes instance to use for colorbar for the azimuth
        color-scale. If None, one is created by taking space from ax
    scan_times
        Scan times of observations as list of 2-tuples containing (start, end)
        times. If None, no scan times are plotted
    savefig
        Full path to save the resulting figure to. If False, figure is not saved
        (default)

    Returns
    -------
    Tuple of matplotlib.figure.Figure, (matplotlib.axes.Axes,
    matplotlib.axes.Axes) which corresponding to (figure, (plot axis, colorbar
    axis)
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, get_sun

    t0_midnight = Time(t0.strftime("%Y-%m-%d 00:00:00.000"),
                       scale='utc',
                       location=(f'{tscop_location.lon.value:.5f}d',
                                 f'{tscop_location.lat.value:.5f}d'))

    if scan_times:
        obs_duration = (max([end for _, end in scan_times]) -
                        t0_midnight).to_value('h')
        dt = np.linspace(0, obs_duration + 24 - (obs_duration % 24),
                         500) * u.hour
    else:
        dt = np.linspace(0, 24, 500) * u.hour

    times = t0_midnight + dt
    dhours = (times - t0_midnight).to('h')
    frame = AltAz(obstime=times, location=tscop_location)
    sun_alts_azs = get_sun(times).transform_to(frame)
    alts_azs = coord_target.transform_to(frame)

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(SUBPLOT_WIDTH_INCHES,
                                              SUBPLOT_WIDTH_INCHES / 1.1))
    else:
        fig = ax.figure

    if cax is None:
        from mpl_toolkits.axes_grid1.axes_size import Fraction, AxesX

        divider = make_axes_locatable(ax)
        cbar_size = Fraction(COLORBAR_WIDTH_FRAC / (1.0 - COLORBAR_WIDTH_FRAC),
                             AxesX(ax))
        cax = divider.append_axes("right", size=cbar_size, pad=0.0)

    points = ax.scatter(dhours.value, alts_azs.alt, c=alts_azs.az,
                        lw=0, s=8, cmap='twilight_shifted', vmin=0., vmax=360.,
                        zorder=3)
    ax.fill_between(dhours.value, 0, 90, sun_alts_azs.alt < -0 * u.deg,
                    color=(0.5, 0.5, 0.5, 0.5), zorder=0)
    ax.fill_between(dhours.value, 0, 90, sun_alts_azs.alt < -18 * u.deg,
                    color=(0, 0, 0, 0.5), zorder=0)

    plt.colorbar(points, cax=cax, ax=ax, orientation='vertical')
    ax.set_xlim(0, np.max(dt.value))
    ax.set_xticks(np.arange(0, np.max(dt.value), 12))
    ax.set_ylim(0, 90)
    ax.set_xlabel(r'$\Delta \left( t - {\rm 00:00UTC} \right) \, '
                  r'\left[ \mathrm{hr} \right]$')
    ax.set_ylabel('Elevation [deg]')

    cax.set_yticks(np.arange(0, 361, 45))
    cax.text(0.5, 0.5, 'Azimuth [deg]', transform=cax.transAxes,
             horizontalalignment='center', verticalalignment='center',
             rotation=90., color='white')

    ax.tick_params(which='both', axis='both', bottom=True, top=True,
                   left=True, right=True, direction='in')
    ax.minorticks_on()
    cax.tick_params(which='both', axis='both', bottom=False, top=False,
                    left=False, right=True, direction='in')
    cax.minorticks_on()

    if scan_times:
        for start, end in scan_times:
            dts = np.linspace((start - t0_midnight).to_value('h'),
                              (end - t0_midnight).to_value('h'), 100)
            dts *= u.hour

            ts = t0_midnight + dts
            frame = AltAz(obstime=ts, location=tscop_location)
            alts_azs = coord_target.transform_to(frame)
            ax.plot(dts.value, alts_azs.alt, ls=':', color='cyan', zorder=3)
            ax.plot(dts[0].value, alts_azs.alt[0].value,
                    marker='o', mec='g', mfc='lawngreen', ms=5, lw=1,
                    zorder=3)
            ax.plot(dts[-1].value, alts_azs.alt[-1].value,
                    marker='o', mec='maroon', mfc='r', ms=5, lw=1,
                    zorder=3)

    if not savefig:
        plt.show()
    else:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, (ax, cax)


def plot_dtec(
        tscop: Telescope, tec_fits: pathlib.Path,
        ax: Optional[matplotlib.pylab.Axes] = None,
        savefig: Union[bool, pathlib.Path] = False
) -> Tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
    """
    Plot the differential total electron content (dTEC) relative to the
    central-most antenna in an array

    Parameters
    ----------
    tscop
        Telescope instance containing array information
    tec_fits
        .fits cube of the TEC-screen with axes (freq, time, y, x)
    ax
        Axes instance to plot to. If not given, a new Axes instance in created
        (default)
    savefig
        If a pathlib.Path is given the figure will be save to it. If False,
        figure will not be saved (default)

    Returns
    -------
    (matplotlib.pylab.Figure, matplotlib.pylab.Axes) instances
    """
    import astropy.units as u
    from astropy.wcs import WCS

    with fits.open(tec_fits) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

    wcs = WCS(hdr)
    dtecs = {k: [] for k, _ in tscop.stations.items()}

    # Get TECs for reference antenna
    _, __, iy, ix = wcs.world_to_array_index_values(
        tscop.stations[tscop.ref_ant].position[0],
        tscop.stations[tscop.ref_ant].position[1], 0 * u.s, 1e8 * u.Hz
    )
    central_tecs = data[0, :, iy, ix]

    for n_station in tscop.stations.keys():
        _, __, iy, ix = wcs.world_to_array_index_values(
            tscop.stations[n_station].position[0],
            tscop.stations[n_station].position[1], 0 * u.s, 1e8 * u.Hz
        )
        dtecs[n_station] = data[0, :, iy, ix] - central_tecs

    # Calculate all times of the TEC .fits cube
    ts = (hdr['CRVAL3'] +
          np.arange(1, hdr['NAXIS3'] + 1) * hdr['CDELT3'] -
          hdr['CRPIX3'] * hdr['CDELT3'])
    ts = (ts - np.min(ts)) / 3600.  # in hours with first time as t = 0hr

    plt.close('all')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(SUBPLOT_WIDTH_INCHES * 2,
                                              SUBPLOT_WIDTH_INCHES))
    else:
        fig = plt.gcf()

    for n_station in tscop.stations.keys():
        ax.plot(ts, dtecs[n_station], ls='-', color=(0, 0, 1, 0.02))

    ax.set_xlim(ts.min(), ts.max())
    ax.set_ylim(-np.max(np.abs(ax.get_ylim())), np.max(np.abs(ax.get_ylim())))

    ax.set_xlabel(r'$t - t_0 \, \left[ \mathrm{hr} \right]$')
    ax.set_ylabel(r'$\mathrm{dTEC} \, \left[ \mathrm{TECU} \right]$')

    ax.minorticks_on()
    ax.tick_params(which='both', axis='both', bottom=True, top=True, left=True,
                   right=True, direction='in')

    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')

    return fig, ax

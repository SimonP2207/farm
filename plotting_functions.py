from typing import Union
import numpy as np
import matplotlib.pylab as plt
import matplotlib.axes
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .classes import SkyModelType
from .image_functions import calculate_spix


def plot_spix(sky_model_type: SkyModelType, precision: float = 0.1,
              ax: Union[None, matplotlib.axes.Axes] = None,
              cax: Union[None, matplotlib.axes.Axes] = None,
              savefig: Union[str, bool] = False):
    spix = calculate_spix(sky_model_type)

    vmin = np.floor(np.nanmin(spix) / precision) * precision
    vmax = np.ceil(np.nanmax(spix) / precision) * precision

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(4, 4 * 1.05))

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.0)

    cmap = cm.get_cmap('jet', 12)
    im = ax.imshow(spix, vmin=vmin, vmax=vmax, cmap=cmap,
                   extent=(+sky_model_type.n_x / 2 * sky_model_type.cdelt,
                           -sky_model_type.n_x / 2 * sky_model_type.cdelt,
                           -sky_model_type.n_y / 2 * sky_model_type.cdelt,
                           +sky_model_type.n_y / 2 * sky_model_type.cdelt))
    plt.colorbar(im, cax=cax, ax=ax, orientation='horizontal')
    cax.xaxis.tick_top()

    ax.set_xticks(range(-4, 5)[::-1])
    ax.set_yticks(range(-4, 5))

    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    ax.tick_params(which='both', axis='both', bottom=True, top=True, left=True,
                   right=True, direction='in')

    ax.set_xlabel(r'$\Delta \, \mathrm{R.A.\, \left[deg\right]}$')
    ax.set_ylabel(r'$\Delta \, \mathrm{Dec.\, \left[deg\right]}$')

    cax.set_xticks(np.arange(vmin, vmax + precision, precision))

    cax.text(0.5, 0.45, r'$\alpha$', transform=cax.transAxes,
             horizontalalignment='center', verticalalignment='center')

    if not savefig:
        plt.show()
    else:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

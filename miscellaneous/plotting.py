"""
Contains all plotting related functionality, matplotlib being data visualisation
tool-of-choice
"""
from typing import Union
import numpy as np
import matplotlib.pylab as plt
import matplotlib.axes
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..sky_model.classes import SkyClassType
from .image_functions import calculate_spix


# def login_required(f):
#     # This function is what we "replace" hello with
#     def wrapper(*args, **kw):
#         args[0].client_session['test'] = True
#         logged_in = 0
#         if logged_in:
#             return f(*args, **kw)  # Call hello
#         else:
#             return redirect(url_for('login'))
#     return wrapper
#
#
# from functools import wraps
# def create_ax_if_needed(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         args_to_kwargs = dict(zip(f.__code__.co_varnames, args))
#         if 'ax' in args_to_kwargs or 'ax' in kwargs:
#
#
#         return f(*args, **kwargs)
#     return wrapper

def plot_spix(sky_model_type: SkyClassType, precision: float = 0.1,
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


def plot_power_spectrum(sky_model_type: SkyClassType, freq: float,
                        ax: Union[None, matplotlib.axes.Axes] = None,
                        savefig: Union[str, bool] = False):
    import powerbox as pbox

    p_k_field, bins_field = pbox.get_power(sky_model_type.t_b(freq),
                                           (0.002, 0.002,))

    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(4, 4 * 1.05))

    ax.plot(bins_field, p_k_field, 'b-')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if not savefig:
        plt.show()
    else:
        fig.savefig(savefig, dpi=300, bbox_inches='tight')

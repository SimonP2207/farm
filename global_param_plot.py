import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def plot_tb_std(fitsfile, astro_param_set):
    number = astro_param_set['number']
    s = ''
    for idx, (k, v) in enumerate(astro_param_set.items()):
        if k == 'number':
            continue
        s += f'{k} = {v:.3f}'
        if idx != len(astro_param_set) - 1:
            s += '\n'

    with fits.open(fitsfile) as hdul:
        hdu = hdul[0]
        data = hdu.data[::-1] * 1000
        hdr = hdu.header

    freqs = [hdr['CRVAL3'] + hdr['CDELT3'] * chan for chan in range(hdr['NAXIS3'])]
    stds = np.nanstd(data, axis=(1, 2))
    tbs = np.nanmean(data, axis=(1, 2))

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(freqs, tbs, color='indianred', ls='-')
    ax.set_xlabel(r'$\nu \left[\mathrm{Hz}\right]$')
    ax2 = ax.twinx()
    ax2.plot(freqs, stds, color='cornflowerblue', ls='-')

    ax.set_ylabel(r"$\bar{T}_\mathrm{b} \, \left[\mathrm{mK}\right]$",
                   color="indianred")

    ax2.set_ylabel(r"$\sigma_\mathrm{std} \, \left[\mathrm{mK}\right]$",
                   color="cornflowerblue")

    ax2.text(0.95, 0.5, s, ha='right', va='center',
             transform=ax2.transAxes, zorder=3)
    ax.tick_params(axis='y', colors='indianred')
    ax2.tick_params(axis='y', colors='cornflowerblue')
    ax2.set_ylim(0, 40)
    ax.set_ylim(-150, 30)
    ax.set_xlim(31299038.34966415, 227374038.34966415)
    ax2.vlines([106e6, 196e6], *ax2.get_ylim(), colors='lightgray',
               linestyles=':')
    fig.savefig(f'global_Tb_param_set_{number}.pdf',
                dpi=150, bbox_inches='tight')

params = {
    'F_STAR10': -1.300,
    'ALPHA_STAR': 0.500,
    'F_ESC10': -1.000,
    'ALPHA_ESC': -0.300,
    'M_TURN': 8.900,
    't_STAR': 0.380,
    'number': 'Eunseong'
    }

eunseong_data = '/Users/simon.purser/pylib/farm/data/files/obs_lightcone_8_512_single_p.fits'
plot_tb_std(eunseong_data, params)

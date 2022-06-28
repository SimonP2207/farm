import copy
import subprocess
import pathlib
from typing import Union, List, Dict, Tuple
from enum import Enum
import numpy as np
from astropy.io import fits
from . import common as sfuncs
from ..miscellaneous import error_handling as errh

text = ('wsclean -weight uniform -taper-gaussian 60 '
        '-super-weight 4 -name image_name -size 8000 8000 '
        '-scale 30asec -channels-out 483 -niter 5000 -pol xx ms_name_ms')


class Option(Enum):
    INCLUDE = True
    EXCLUDE = False


_WSCLEAN_DEFAULT_ARGS = {
    # General Options
    'j': 1,
    'parallel-gridding': 1,
    'parallel-reordering': 1,
    'mem': 100.,
    'abs-mem': None,
    'direct-allocation': Option.EXCLUDE,
    'verbose': Option.INCLUDE,
    'log-time': Option.INCLUDE,
    'quiet': Option.EXCLUDE,
    'reorder': Option.INCLUDE,
    'no-reorder': Option.EXCLUDE,
    'temp-dir': None,
    'update-model-required': Option.INCLUDE,
    'no-update-model-required': Option.EXCLUDE,
    'no-dirty': Option.EXCLUDE,
    'save-first-residual': Option.EXCLUDE,
    'save-weights': Option.INCLUDE,
    'save-uv': Option.EXCLUDE,
    'reuse-psf': None,
    'reuse-dirty': None,
    'apply-primary-beam': Option.EXCLUDE,
    'reuse-primary-beam': Option.EXCLUDE,
    'use-differential-lofar-beam': Option.EXCLUDE,
    'mwa-path': None,
    'save-psf-pb': Option.EXCLUDE,
    'pb-undesampling': 8.,
    'dry-run': Option.EXCLUDE,
    # Weighting Options
    'weight': "uniform",
    'super-weight': 1.0,
    'mf-weighting': Option.EXCLUDE,
    'no-mf-weighting': Option.INCLUDE,
    'weighting-rank-filter': None,
    'weighting-rank-filter-size': 16.,
    'taper-gaussian': None,
    'taper-tukey': None,
    'taper-inner-tukey': None,
    'taper-edge': None,
    'taper-edge-tukey': None,
    'use-weights-as-taper': Option.EXCLUDE,
    'store-imaging-weights': Option.INCLUDE,
    # Inversion Options
    'name': 'wsclean',
    'size': None,  # <width> <height>
    'padding': 1.2,
    'scale': '0.01deg',
    'predict': Option.EXCLUDE,
    'continue': Option.EXCLUDE,
    'subtract-model': Option.EXCLUDE,
    'channels-out': 1,
    'gap-channel-division': Option.EXCLUDE,
    'channel-division-frequencies': None,
    'nwlayers': None,
    'nwlayers-factor': None,
    'nwlayers-for-size': None,
    'no-small-inversion': Option.EXCLUDE,
    'small-inversion': Option.INCLUDE,
    'grid-mode': "kb",
    'kernel-size': 7,
    'oversampling': 63,
    'make-psf': Option.INCLUDE,
    'make-psf-only': Option.EXCLUDE,
    'visibility-weighting-mode': 'normal',
    'no-normalize-for-weighting': Option.EXCLUDE,
    'baseline-averaging': None,
    'simulate-noise': None,
    'direct-ft': Option.EXCLUDE,
    'use-idg': Option.EXCLUDE,
    'idg-mode': 'cpu',
    'use-wgridder': Option.EXCLUDE,
    # A-Term Gridding
    'aterm-config': None,
    'grid-with-beam': Option.EXCLUDE,
    'beam-aterm-update': 300.,
    'aterm-kernel-size': Option.EXCLUDE,
    'save-aterms': Option.EXCLUDE,
    # Data Selection Options
    'pol': 'I',
    'interval': None,
    'intervals-out': 1,
    'even-timesteps': Option.EXCLUDE,
    'odd-timesteps': Option.EXCLUDE,
    'channel-range': None,
    'field': 0,
    'spws': None,  # Comma-separated list of integers
    'data-column': 'CORRECTED_DATA',
    'maxuvw-m': None,
    'minuvw-m': None,
    'maxuv-l': None,
    'minuv-l': None,
    'maxw': 100.,
    # Deconvolution Options
    'niter': 0,
    'nmiter': 20,
    'threshold': 0.0,
    'auto-threshold': None,
    'auto-mask': None,
    'local-rms': Option.EXCLUDE,
    'local-rms-window': 25,
    'local-rms-method': 'rms',
    'gain': 0.1,
    'mgain': 1.0,
    'join-polarizations': Option.EXCLUDE,
    'link-polarizations': None,
    'join-channels': Option.EXCLUDE,
    'spectral-correction': None,
    'no-fast-subminor': Option.INCLUDE,
    'multiscale': Option.EXCLUDE,
    'multiscale-scale-bias': 0.6,
    'multiscale-scales': None,
    'multiscale-shape': 'tapered-quadratic',
    'multiscale-gain': 0.1,
    'multiscale-convolution-padding': 1.1,
    'no-multiscale-fast-subminor': Option.INCLUDE,
    'iuwt': Option.EXCLUDE,
    'iuwt-snr-test': Option.EXCLUDE,
    'no-iuwt-snr-test': Option.INCLUDE,
    'moresane-ext': None,
    'moresane-arg': None,
    'moresane-sl': None,
    'save-source-list': Option.EXCLUDE,
    'clean-border': 0.,
    'fits-mask': None,
    'casa-mask': None,
    'horizon-mask': None,
    'no-negative': Option.EXCLUDE,
    'negative': Option.INCLUDE,
    'stop-negative': Option.EXCLUDE,
    'fits-spectral-pol': None,  # <nterms>
    'fit-spectral-log-pol': None,  # <nterms>
    'deconvolution-channels': None,  # <nchannels>
    'squared-channel-joining': Option.EXCLUDE,
    'parallel-deconvolution': None,  # <maxsize>
    # Restoration Options
    'restore': None,  # <input residual> <input model> <output image>
    'beam-size': None,  # <arcsec>
    'beam-shape': None,  # <maj in arcsec> <min in arcsec> <position angle in deg>
    'fit-beam': Option.INCLUDE,
    'no-fit-beam': Option.EXCLUDE,
    'beam-fitting-size': None,  # <factor>
    'theoretic-beam': Option.EXCLUDE,
    'circular-beam': Option.EXCLUDE,
    'elliptical-beam': Option.INCLUDE
}


def wsclean(ms_list: Union[Union[str, pathlib.Path],
                           List[Union[str, pathlib.Path]]],
            kw_args: Dict,
            consolidate_channels: bool = True,
            dryrun: bool = False) -> Tuple[str, Dict]:
    """
    Executes wsclean on measurement set(s) according to specified kwargs

    Parameters
    ----------
    ms_list
        Path, or list of paths, to measurement sets
    kw_args
        Dictionary containing keys for wsclean command-line arguments (without
        preceding '-') whose values are those given next to that wsclean
        argument on the command-line. Each kwarg should be a command-line option
        for wsclean to run. Only bool, str, float and int instances are accepted
        for values. If a wsclean argument requires a list, format that list as
        would be required on the command line
        e.g. beam-shape: '1amin 1amin 3deg', multiscale-scales: '4, 20, 50, 100'
        If an option needs to be switched on/off (i.e. no values on the
        command-line, then use a bool to denote whether to include that option
        or not
    consolidate_channels
        If 'channels-out' is specified in kw_args and its value is > 1, wsclean
        outputs separate .fits images for each channel. In the case you want a
        single .fits image cube, set this to True (default)
    dryrun
        Whether to return the command-line command for wsclean without running
        wsclean. Default is False. Mainly for testing purposes

    Return
    ------
    Tuple of command-line command for running wsclean and products as the
    2-tuple (str, dict)

    Raises
    ------
    KeyError
        If a key in the kw_args dict is not recognised as a wsclean command-line
        argument
    ValueError
        If an empty dictionary is passed as kw_args
    """
    if not kw_args:
        errh.raise_error(ValueError, "kw_args can not be empty")

    # Remove
    input_args = copy.deepcopy(_WSCLEAN_DEFAULT_ARGS)
    for k, v in kw_args.items():
        if k not in _WSCLEAN_DEFAULT_ARGS:
            errh.raise_error(KeyError,
                             f"'{k}' not a valid wsclean command-line arg")

        if isinstance(_WSCLEAN_DEFAULT_ARGS[k], Option):
            if _WSCLEAN_DEFAULT_ARGS[k].value != v:
                input_args[k] = Option.INCLUDE if v else Option.EXCLUDE
            else:
                input_args.pop(k)
        elif v != _WSCLEAN_DEFAULT_ARGS[k]:
            input_args[k] = v
        else:
            input_args.pop(k)

    # Remove wsclean arguments that aren't mentioned in kw_args (and thereby
    # have their default values)
    for k in _WSCLEAN_DEFAULT_ARGS:
        if k not in kw_args:
            input_args.pop(k)

    cmd = f"{sfuncs.which('wsclean')}"
    for k, v in input_args.items():
        if isinstance(v, Option):
            cmd += f' -{k}'
        else:
            cmd += f' -{k} {v}'

    if isinstance(ms_list, (str, pathlib.Path)):
        cmd += f' {ms_list}'
    else:
        cmd += ' ' + ' '.join(ms_list)

    output_name = 'wsclean' if 'name' not in kw_args else kw_args['name']
    output_name = pathlib.Path(output_name)
    output_dcy = output_name.parent.resolve()
    output_prefix = output_name.name
    wsclean_products = output_dcy.glob(f'./{output_prefix}*.fits')

    for wsclean_product in wsclean_products:
        wsclean_product.unlink()

    if not dryrun:
        subprocess.run(cmd, shell='True')

    products = {
        'dirty': [], 'image': [], 'model': [], 'psf': [], 'residual': []
    }
    for wsclean_product in output_dcy.glob(f'./{output_prefix}*.fits'):
        for k in products:
            if k in wsclean_product.name:
                products[k].append(wsclean_product)
    products = {k: sorted(v) for k, v in products.items()}

    if all([len(value) in (0, 1) for value in products.values()]):
        for im_type in products:
            rename = pathlib.Path(
                output_dcy / f'{output_prefix}-{im_type}.fits'
            )
            if len(products[im_type]) > 0:
                products[im_type][0].rename(rename)
                products[im_type][0] = rename

        return cmd, products

    if consolidate_channels:
        for im_type in products:
            rename = pathlib.Path(
                output_dcy / f'{output_prefix}-{im_type}.fits'
            )
            with fits.open(products['image'][0]) as hdul:
                im_hdr = hdul[0].header

            im_data = np.zeros((len(products[im_type]),
                                im_hdr['NAXIS2'],
                                im_hdr['NAXIS1']), dtype=np.float32)

            for ichan, image in enumerate(products[im_type]):
                with fits.open(image) as hdul:
                    im_data[ichan, :, :] = hdul[0].data[:, :]

            hdu_cube = fits.PrimaryHDU(im_data)
            hdu_cube.header = im_hdr
            hdu_cube.writeto(rename)

            for im in products[im_type]:
                im.unlink()

            products[im_type] = rename

    return cmd, products

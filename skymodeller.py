"""
Sky model related functions/classes
"""
from farm import LOGGER


def make_sky_model(sky0, settings, radius_deg, flux_min_outer_jy):
    """Filter sky model.
    Includes all sources within the given radius, and sources above the
    specified flux outside this radius.
    """
    # Get pointing centre.
    ra0_deg = float(settings['observation/phase_centre_ra_deg'])
    dec0_deg = float(settings['observation/phase_centre_dec_deg'])
    # Create "inner" and "outer" sky models.
    sky_inner = sky0.create_copy()
    sky_outer = sky0.create_copy()
    sky_inner.filter_by_radius(0.0, radius_deg, ra0_deg, dec0_deg)
    sky_outer.filter_by_radius(radius_deg, 180.0, ra0_deg, dec0_deg)
    sky_outer.filter_by_flux(flux_min_outer_jy, 1e9)
    LOGGER.info("Number of sources in sky0: %d", sky0.num_sources)
    LOGGER.info("Number of sources in inner sky model: %d",
                 sky_inner.num_sources)
    LOGGER.info("Number of sources in outer sky model above %.3f Jy: %d",
                 flux_min_outer_jy, sky_outer.num_sources)
    sky_outer.append(sky_inner)
    LOGGER.info("Number of sources in output sky model: %d",
                 sky_outer.num_sources)
    return sky_outer

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Colour Models Utilities
==============================

Defines various colour models common utilities.
"""

from __future__ import division, unicode_literals

from colour.models import (
    Lab_to_LCHab,
    Luv_to_LCHuv,
    Luv_to_uv,
    UCS_to_uv,
    XYZ_to_IPT,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_UCS,
    XYZ_to_UVW,
    XYZ_to_xy,
    XYZ_to_xyY)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['COLOURSPACE_MODELS',
           'COLOURSPACE_MODELS_LABELS',
           'XYZ_to_colourspace_model']

COLOURSPACE_MODELS = (
    'CIE XYZ',
    'CIE xyY',
    'CIE Lab',
    'CIE Luv',
    'CIE UCS',
    'CIE UVW',
    'IPT')

COLOURSPACE_MODELS_LABELS = {
    'CIE XYZ': ('X', 'Y', 'Z'),
    'CIE xyY': ('x', 'y', 'Y'),
    'CIE Lab': ('$a^*$', '$b^*$', '$L^*$'),
    'CIE Luv': ('$u^\prime$', '$v^\prime$', '$L^*$'),
    'CIE UCS': ('U', 'V', 'W'),
    'CIE UVW': ('U', 'V', 'W'),
    'IPT': ('P', 'T', 'I')}

"""
Colourspace models labels mapping.

COLOURSPACE_MODELS_LABELS : dict
    **{'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE Luv', 'CIE UCS', 'CIE UVW',
    'IPT'}**
"""


def XYZ_to_colourspace_model(XYZ, illuminant, model):
    """
    Converts from *CIE XYZ* tristimulus values to given colourspace model.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like
        *CIE XYZ* tristimulus values *illuminant* *xy* chromaticity
        coordinates.
    model : unicode
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE Luv', 'CIE Luv uv',
        'CIE UCS', 'CIE UCS uv', 'CIE UVW', 'IPT'}**,
        Colourspace model to convert the *CIE XYZ* tristimulus values to.

    Returns
    -------
    ndarray
        Colourspace model values.

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> W = np.array([0.34567, 0.35850])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE XYZ')
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xyY')
    array([ 0.2641477...,  0.3777000...,  0.1008    ])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE xy')
    array([ 0.2641477...,  0.3777000...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Lab')
    array([ 37.9856291..., -23.6230288...,  -4.4141703...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHab')
    array([  37.9856291...,   24.0319036...,  190.5841597...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv')
    array([ 37.9856291..., -28.7922944...,  -1.3558195...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE Luv uv')
    array([ 0.1508531...,  0.4853297...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE LCHuv')
    array([  37.9856291...,   28.8241993...,  182.6960474...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UCS uv')
    array([ 0.1508531...,  0.32355314...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'CIE UVW')
    array([-28.0483277...,  -0.8805242...,  37.0041149...])
    >>> XYZ_to_colourspace_model(  # doctest: +ELLIPSIS
    ... XYZ, W, 'IPT')
    array([ 0.3657112..., -0.1111479...,  0.0159474...])
    """

    values = None
    if model == 'CIE XYZ':
        values = XYZ
    if model == 'CIE xyY':
        values = XYZ_to_xyY(XYZ, illuminant)
    if model == 'CIE xy':
        values = XYZ_to_xy(XYZ, illuminant)
    if model == 'CIE Lab':
        values = XYZ_to_Lab(XYZ, illuminant)
    if model == 'CIE LCHab':
        values = Lab_to_LCHab(XYZ_to_Lab(XYZ, illuminant))
    if model == 'CIE Luv':
        values = XYZ_to_Luv(XYZ, illuminant)
    if model == 'CIE Luv uv':
        values = Luv_to_uv(XYZ_to_Luv(XYZ, illuminant), illuminant)
    if model == 'CIE LCHuv':
        values = Luv_to_LCHuv(XYZ_to_Luv(XYZ, illuminant))
    if model == 'CIE UCS':
        values = XYZ_to_UCS(XYZ)
    if model == 'CIE UCS uv':
        values = UCS_to_uv(XYZ_to_UCS(XYZ))
    if model == 'CIE UVW':
        values = XYZ_to_UVW(XYZ * 100, illuminant)
    if model == 'IPT':
        values = XYZ_to_IPT(XYZ)

    if values is None:
        raise ValueError(
            '"{0}" not found in colourspace models: "{1}".'.format(
                model, ', '.join(COLOURSPACE_MODELS)))

    return values

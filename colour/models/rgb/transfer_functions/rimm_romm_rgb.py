#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RIMM / ROMM / ERIMM Encodings OETF (OECF) and EOTF (EOCF)
=========================================================

Defines the *RIMM / ROMM / ERIMM* encodings OETF (OECF) and EOTF (EOCF):

-   :func:`oetf_ROMMRGB`
-   :func:`eotf_ROMMRGB`
-   :func:`oetf_ProPhotoRGB`
-   :func:`eotf_ProPhotoRGB`
-   :func:`oetf_RIMMRGB`
-   :func:`eotf_RIMMRGB`
-   :func:`log_encoding_ERIMMRGB`
-   :func:`log_decoding_ERIMMRGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Spaulding, K. E., Woolfe, G. J., & Giorgianni, E. J. (2000). Reference
        Input/Output Medium Metric RGB Color Encodings (RIMM/ROMM RGB), 1–8.
        Retrieved from http://www.photo-lovers.org/pdf/color/romm.pdf
.. [3]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['oetf_ROMMRGB',
           'eotf_ROMMRGB',
           'oetf_ProPhotoRGB',
           'eotf_ProPhotoRGB',
           'oetf_RIMMRGB',
           'eotf_RIMMRGB',
           'log_encoding_ERIMMRGB',
           'log_decoding_ERIMMRGB']


def oetf_ROMMRGB(value):
    """
    Defines the *ROMM RGB* encoding opto-electronic transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> oetf_ROMMRGB(0.18)  # doctest: +ELLIPSIS
    0.3857114...
    """

    value = np.asarray(value)

    return as_numeric(np.where(value < 0.001953,
                               value * 16,
                               value ** (1 / 1.8)))


def eotf_ROMMRGB(value):
    """
    Defines the *ROMM RGB* encoding electro-optical transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> eotf_ROMMRGB(0.3857114247511376)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    return as_numeric(np.where(
        value < oetf_ROMMRGB(0.001953),
        value / 16,
        value ** 1.8))


oetf_ProPhotoRGB = oetf_ROMMRGB
eotf_ProPhotoRGB = eotf_ROMMRGB


def oetf_RIMMRGB(value, I_max=255, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding opto-electronic transfer function.

    *RIMM RGB* encoding non-linearity is based on that specified by
    *Recommendation ITU-R BT.709-6*.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    I_max : numeric, optional
        Maximum code value: 255, 4095, and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> oetf_RIMMRGB(0.18)  # doctest: +ELLIPSIS
    74.3768017...
    """

    value = np.asarray(value)

    V_clip = 1.099 * E_clip ** 0.45 - 0.099
    q = I_max / V_clip

    X_p_RIMM = np.select(
        [value < 0.0,
         value < 0.018, value >= 0.018,
         value > E_clip],
        [0, 4.5 * value, 1.099 * (value ** 0.45) - 0.099, I_max])

    return as_numeric(q * X_p_RIMM)


def eotf_RIMMRGB(value, I_max=255, E_clip=2.0):
    """
    Defines the *RIMM RGB* encoding electro-optical transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    I_max : numeric, optional
        Maximum code value: 255, 4095, and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_clip : numeric, optional
        Maximum exposure level.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> eotf_RIMMRGB(74.37680178131521)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    V_clip = 1.099 * E_clip ** 0.45 - 0.099

    m = V_clip * value / I_max

    X_RIMM = np.where(
        value < oetf_RIMMRGB(0.018),
        m / 4.5, ((m + 0.099) / 1.099) ** (1 / 0.45))

    return as_numeric(X_RIMM)


def log_encoding_ERIMMRGB(value,
                          I_max=255,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    I_max : numeric, optional
        Maximum code value: 255, 4095, and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_ERIMMRGB(0.18)  # doctest: +ELLIPSIS
    104.5633593...
    """

    value = np.asarray(value)

    E_t = np.exp(1) * E_min

    X_p = np.select(
        [value < 0.0,
         value <= E_t, value > E_t,
         value > E_clip],
        [0,
         I_max * ((np.log(E_t) - np.log(E_min)) /
                  (np.log(E_clip) - np.log(E_min))) * (value / E_t),
         I_max * ((np.log(value) - np.log(E_min)) /
                  (np.log(E_clip) - np.log(E_min))),
         I_max])

    return as_numeric(X_p)


def log_decoding_ERIMMRGB(value,
                          I_max=255,
                          E_min=0.001,
                          E_clip=316.2):
    """
    Defines the *ERIMM RGB* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    I_max : numeric, optional
        Maximum code value: 255, 4095, and 650535 for respectively 8-bit,
        12-bit and 16-bit per channel.
    E_min : numeric, optional
        Minimum exposure limit.
    E_clip : numeric, optional
        Maximum exposure limit.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_ERIMMRGB(104.56335932049294)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    E_t = np.exp(1) * E_min

    X = np.where(
        value <= I_max * ((np.log(E_t) - np.log(E_min)) /
                          (np.log(E_clip) - np.log(E_min))),
        (((np.log(E_clip) - np.log(E_min)) / (np.log(E_t) - np.log(E_min))) *
         ((value * E_t) / I_max)),
        np.exp((value / I_max) *
               (np.log(E_clip) - np.log(E_min)) + np.log(E_min)))

    return as_numeric(X)

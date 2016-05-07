#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sRGB Colourspace
================

Defines the *sRGB* colourspace:

-   :attr:`sRGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Electrotechnical Commission. (1999). IEC 61966-2-1:1999 -
        Multimedia systems and equipment - Colour measurement and management -
        Part 2-1: Colour management - Default RGB colour space - sRGB, 51.
        Retrieved from https://webstore.iec.ch/publication/6169
.. [2]  International Telecommunication Union. (2002). Parameter values for
        the HDTV standards for production and international programme exchange
        BT Series Broadcasting service. In Recommendation ITU-R BT.709-6
        (Vol. 5, pp. 1–32). ISBN:9519982000
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_sRGB,
    eotf_sRGB,
    normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['sRGB_PRIMARIES',
           'sRGB_ILLUMINANT',
           'sRGB_WHITEPOINT',
           'sRGB_TO_XYZ_MATRIX',
           'XYZ_TO_sRGB_MATRIX',
           'sRGB_COLOURSPACE']

sRGB_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])
"""
*sRGB* colourspace primaries.

sRGB_PRIMARIES : ndarray, (3, 2)
"""

sRGB_ILLUMINANT = 'D65'
"""
*sRGB* colourspace whitepoint name as illuminant.

sRGB_WHITEPOINT : unicode
"""

sRGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(sRGB_ILLUMINANT)
"""
*sRGB* colourspace whitepoint.

sRGB_WHITEPOINT : tuple
"""

sRGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    sRGB_PRIMARIES, sRGB_WHITEPOINT)
"""
*sRGB* colourspace to *CIE XYZ* tristimulus values matrix.

sRGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_sRGB_MATRIX = np.linalg.inv(sRGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *sRGB* colourspace matrix.

XYZ_TO_sRGB_MATRIX : array_like, (3, 3)
"""

sRGB_COLOURSPACE = RGB_Colourspace(
    'sRGB',
    sRGB_PRIMARIES,
    sRGB_WHITEPOINT,
    sRGB_ILLUMINANT,
    sRGB_TO_XYZ_MATRIX,
    XYZ_TO_sRGB_MATRIX,
    oetf_sRGB,
    eotf_sRGB)
"""
*sRGB* colourspace.

sRGB_COLOURSPACE : RGB_Colourspace
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.photometry` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.colorimetry import (
    ILLUMINANTS_RELATIVE_SPDS,
    LIGHT_SOURCES_RELATIVE_SPDS,
    luminous_flux,
    luminous_efficiency,
    luminous_efficacy,
    zeros_spd)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLuminousFlux',
           'TestLuminousEfficiency',
           'TestLuminousEfficacy']


class TestLuminousFlux(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_flux` definition unit
    tests methods.
    """

    def test_luminous_flux(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_flux` definition.
        """

        self.assertAlmostEqual(
            luminous_flux(
                ILLUMINANTS_RELATIVE_SPDS.get('F2').clone().normalise()),
            28588.736129772711,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'Neodimium Incandescent')),
            23807.655527367198,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'F32T8/TL841 (Triphosphor)')),
            13090.067590531509,
            places=7)


class TestLuminousEfficiency(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_efficiency`
    definition unit tests methods.
    """

    def test_luminous_efficiency(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_efficiency`
        definition.
        """

        self.assertAlmostEqual(
            luminous_efficiency(
                ILLUMINANTS_RELATIVE_SPDS.get('F2').clone().normalise()),
            0.493176239758,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'Neodimium Incandescent')),
            0.199439356245,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'F32T8/TL841 (Triphosphor)')),
            0.510809188121,
            places=7)


class TestLuminousEfficacy(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_efficacy`
    definition unit tests methods.
    """

    def test_luminous_efficacy(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_efficacy`
        definition.
        """

        self.assertAlmostEqual(
            luminous_efficacy(
                ILLUMINANTS_RELATIVE_SPDS.get('F2').clone().normalise()),
            336.83937175510914,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'Neodimium Incandescent')),
            136.21708031547874,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_RELATIVE_SPDS.get(
                'F32T8/TL841 (Triphosphor)')),
            348.88267548693346,
            places=7)

        spd = zeros_spd()
        spd[555] = 1
        self.assertAlmostEqual(
            luminous_efficacy(spd),
            682.99999999999989,
            places=7)


if __name__ == '__main__':
    unittest.main()

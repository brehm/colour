#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases corresponding colour blindness plotting examples.
"""

import numpy as np
import os

import colour
from colour.plotting import *  # noqa
from colour.utilities.verbose import message_box

RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resources')

# Notes
# -----
# - `cvd_simulation_Machado2010_plot` expects *RGB* linear values.

INVERSE_OECF = colour.RGB_COLOURSPACES['sRGB'].inverse_transfer_function
ISHIHARA_CBT_3_IMAGE = INVERSE_OECF(colour.read_image(os.path.join(
    RESOURCES_DIRECTORY, 'Ishihara_Colour_Blindness_Test_Plate_3.png')))

message_box('Colour Blindness Plots')

message_box('Displaying "Ishihara Colour Blindness Test - Plate 3".')
image_plot(ISHIHARA_CBT_3_IMAGE, 'Normal Trichromat', label_colour='black')

print('\n')

message_box('Simulating average "Protanomaly" on '
            '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) '
            'model and pre-computed matrix.')
cvd_simulation_Machado2010_plot(
    ISHIHARA_CBT_3_IMAGE, 'Protanomaly', 0.5, label_colour='black')

print('\n')

M_a = colour.anomalous_trichromacy_matrix_Machado2010(
    colour.LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals'),
    colour.DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997'],
    np.array([10, 0, 0]))
label = 'Average Protanomaly - 10nm'
message_box('Simulating average "Protanomaly" on '
            '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) '
            'model using "Stockman & Sharpe 2 Degree Cone Fundamentals" and '
            '"Typical CRT Brainard 1997" "RGB" display primaries.')
cvd_simulation_Machado2010_plot(
    ISHIHARA_CBT_3_IMAGE, M_a=M_a, label=label, label_colour='black')

print('\n')

M_a = colour.anomalous_trichromacy_matrix_Machado2010(
    colour.LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals'),
    colour.DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997'],
    np.array([20, 0, 0]))
label = 'Protanopia - 20nm'
message_box('Simulating "Protanopia" on '
            '"Ishihara Colour Blindness Test - Plate 3" with Machado (2010) '
            'model using "Stockman & Sharpe 2 Degree Cone Fundamentals" and '
            '"Typical CRT Brainard 1997" "RGB" display primaries.')
cvd_simulation_Machado2010_plot(
    ISHIHARA_CBT_3_IMAGE, M_a=M_a, label=label, label_colour='black')

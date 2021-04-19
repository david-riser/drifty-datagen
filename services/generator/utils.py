"""
File: services/generator/utils.py
Author: David Riser
Date Created: 4/19/2021
Date Modified: 4/19/2021
Purpose: Utility functions required for this
         application.
"""
from pathlib import Path

import numpy as np


def check_and_create(target):
    """ Use pathlib to check and create the output
    folder if it does not exist.
    """
    output_path = Path(target)
    if not output_path.exists():
        output_path.mkdir(parents=True)


def load_generation_params(config):
    """ From the provided configuration file
    load the parameters that define class
    location, shape, and evolution. """

    centers = np.array([
        [config["generator"]["center_x1"], config["generator"]["center_y1"]],
        [config["generator"]["center_x2"], config["generator"]["center_y2"]]
    ])
    stds = np.array([
        [config["generator"]["std_x1"], config["generator"]["std_y1"]],
        [config["generator"]["std_x2"], config["generator"]["std_y2"]]
    ])
    jumps = np.array([
        [config["generator"]["jump_x1"], config["generator"]["jump_y1"]],
        [config["generator"]["jump_x2"], config["generator"]["jump_y2"]]
    ])

    return centers, stds, jumps

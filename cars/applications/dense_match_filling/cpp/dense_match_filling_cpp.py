"""
this module contains the headers of the dense_match_filling_cpp module.
"""

# pylint: skip-file

from typing import Tuple

import numpy as np


def fill_disp_pandora(
    disp: np.ndarray, msk_fill_disp: np.ndarray, nb_directions: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolation of the left disparity map to fill holes.
    Interpolate invalid pixel by finding the nearest correct pixels in
    8/16 different directions and use the median of their disparities.
    ?bontar, J., & LeCun, Y. (2016). Stereo matching by training
    a convolutional neural network to compare image
    patches. The journal of machine learning research, 17(1), 2287-2318.
    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching
    and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence,
    2007, vol. 30, no 2, p. 328-341.

    Copied/adapted fct from pandora/validation/interpolated_disparity.py

    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param msk_fill_disp: validity mask
    :type msk_fill_disp: 2D np.array (row, col)
    :param nb_directions: nb directions to explore
    :type nb_directions: integer

    :return: the interpolate left disparity map,
        with the validity mask update :
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    return None, None


def find_valid_neighbors(
    dirs: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    row: int,
    col: int,
    nb_directions: int,
):
    """
    Find valid neighbors along directions

    Copied/adapted fct from pandora/validation/interpolated_disparity.py

    :param dirs: directions
    :type dirs: 2D np.array (row, col)
    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param row: row current value
    :type row: int
    :param col: col current value
    :type col: int
    :param nb_directions: nb directions to explore
    :type nb_directions: int

    :return: valid neighbors
    :rtype: 2D np.array
    """
    ...

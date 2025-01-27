"""
this module contains the headers of the dense_matching_cpp module.
"""

# pylint: skip-file


def estimate_right_classif_on_left(
    right_classif, disp_map, disp_mask, disp_min, disp_max
):
    """
    Estimate right classif on left image

    :param right_classif: right classification
    :type right_classif: np ndarray
    :param disp_map: disparity map
    :type disp_map: np ndarray
    :param disp_mask: disparity mask
    :type disp_mask: np ndarray
    :param disp_min: disparity min
    :type disp_min: int
    :param disp_max: disparity max
    :type disp_max: int

    :return: right classif on left image
    :rtype: np nadarray
    """
    ...


def mask_left_classif_from_right_mask(
    left_classif, right_mask, disp_min, disp_max
):
    """
    Mask left classif with right mask.

    :param left_classif: right classification
    :type left_classif: np ndarray
    :param right_mask: right mask
    :type right_mask: np ndarray
    :param disp_min: disparity min
    :type disp_min: np.array type int
    :param disp_max: disparity max
    :type disp_max: np.array type int

    :return: masked left classif
    :rtype: np nadarray
    """
    ...


def estimate_right_grid_disp(disp_min_grid, disp_max_grid):
    """
    Estimate right grid min and max.
    Correspond to the range of pixels that can be correlated
    from left -> right.
    If no left pixels can be associated to right, use global values

    :param disp_min_grid: left disp min grid
    :type disp_min_grid: numpy ndarray
    :param disp_max_grid: left disp max grid
    :type disp_max_grid: numpy ndarray

    :return: disp_min_right_grid, disp_max_right_grid
    :rtype: numpy ndarray, numpy ndarray
    """
    return None, None

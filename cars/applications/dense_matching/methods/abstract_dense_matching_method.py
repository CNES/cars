"""
This module contains the abstract dense matching method class
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

import xarray as xr


class AbstractDenseMatchingMethod(metaclass=ABCMeta):
    """
    AbstractDenseMatchingMethod
    This class is an abstraction for the low-level dense matching method concept
    """

    available_methods: Dict = {}
    default_method = "pandora"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required method
        :raises:
         - KeyError when the required method is not registered

        :param conf: configuration for matching
        :return: a method_to_use object
        """

        matching_method = cls.default_method
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Dense Matching method not specified, "
                "default {} is used".format(matching_method)
            )
        else:
            matching_method = conf.get("method", cls.default_method)

        if matching_method not in cls.available_methods:
            logging.error(
                "No matching method named {} registered".format(matching_method)
            )
            raise KeyError(
                "No matching method named {} registered".format(matching_method)
            )

        logging.info(
            "The AbstractDenseMatchingMethod({}) method will be used".format(
                matching_method
            )
        )

        return super(AbstractDenseMatchingMethod, cls).__new__(
            cls.available_methods[matching_method]
        )

    def __init_subclass__(cls, short_name, **kwargs):  # pylint: disable=E0302
        super().__init_subclass__(**kwargs)
        for name in short_name:
            cls.available_methods[name] = cls

    def __init__(self, conf=None):  # pylint: disable=W0613
        # attributes defined in implementations
        self.schema = None
        self.used_config = None
        self.loader = None

    @abstractmethod
    def get_optimal_tile_size(
        self,
        disp_range_grid,
        max_ram_per_worker,
        min_epi_tile_size,
        max_epi_tile_size,
        local_disp_grid_step,
        epipolar_tile_margin_in_percent,
    ):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_range_grid: minimum and maximum disparity grid
        :param max_ram_per_worker: maximum ram per worker
        :return: optimal tile size

        """

    @abstractmethod
    def get_method(self):
        """
        Returns the method used
        """

    @abstractmethod
    def get_performance_map_parameters(self):
        """
        Get parameter linked to performance, that will be used in triangulation

        :return: parameters to use
        :type: dict
        """

    @abstractmethod
    def get_margins_fun(
        self,
        grid_left,
        disp_range_grid,
        min_elevation_offset,
        max_elevation_offset,
    ):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling

        :param grid_left: left epipolar grid
        :type grid_left: dict
        :param disp_range_grid: minimum and maximum disparity grid
        :return: function that generates margin for given roi
        """

    @abstractmethod
    def run(
        self,
        left_image_object: xr.Dataset,
        right_image_object: xr.Dataset,
        disp_range_grid,
        saving_info=None,
        compute_disparity_masks=False,
        crop_with_range=None,
        left_overlaps=None,
        margins_to_keep=0,
        texture_bands=None,
        classif_bands_to_mask=None,
    ):
        """
        Run dense matching on a pair of epipolar images.

        Compute the disparity map between left and right images using the
        configured dense matching method.

        :param left_image_object: left epipolar image as xarray Dataset with:
                - data with keys: "im", optionally "msk", "texture", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max",
                "transform", "crs", "valid_pixels", "no_data_mask",
                "no_data_img"
        :type left_image_object: xr.Dataset
        :param right_image_object: right epipolar image as xarray Dataset with:
                - data with keys: "im", optionally "msk", "texture", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max",
                "transform", "crs", "valid_pixels", "no_data_mask",
                "no_data_img"
        :type right_image_object: xr.Dataset
        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param saving_info: information for saving outputs
        :type saving_info: dict
        :param compute_disparity_masks: activate computation of disparity masks
        :type compute_disparity_masks: bool
        :param crop_with_range: crop disparity map using provided range
        :type crop_with_range: int or None
        :param left_overlaps: overlaps associated to left image tiles
        :type left_overlaps: dict or None
        :param margins_to_keep: margins to keep after dense matching
        :type margins_to_keep: int
        :param texture_bands: indices of bands from left image used for texture
        :type texture_bands: list
        :param classif_bands_to_mask: bands from classification to mask
        :type classif_bands_to_mask: list of str or int

        :return: disparity map: \
            The CarsDataset contains:
            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                 "disp_min_tiling", "disp_max_tiling"

        :rtype: CarsDataset
        """

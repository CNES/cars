"""
This module contains the abstract sparse matching method class
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

import xarray as xr


class AbstractSparseMatchingMethod(metaclass=ABCMeta):
    """
    AbstractSparseMatchingMethod
    This class is an abstraction for the low-level sparse matching method
    concept.
    """

    available_methods: Dict = {}
    default_method = "sift"

    def __new__(cls, conf=None):  # pylint: disable=W0613
        """
        Return the required method.

        :raises:
         - KeyError when the required method is not registered

        :param conf: configuration for matching
        :return: a method_to_use object
        """

        matching_method = cls.default_method
        if bool(conf) is False or "method" not in conf:
            logging.info(
                "Sparse Matching method not specified, default "
                "{} is used".format(matching_method)
            )
        else:
            matching_method = conf.get("method", cls.default_method)

        if matching_method not in cls.available_methods:
            logging.error(
                "No sparse matching method named {} registered".format(
                    matching_method
                )
            )
            raise KeyError(
                "No sparse matching method named {} registered".format(
                    matching_method
                )
            )

        logging.info(
            "The AbstractSparseMatchingMethod({}) method will be used".format(
                matching_method
            )
        )

        return super(AbstractSparseMatchingMethod, cls).__new__(
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

    @abstractmethod
    def get_required_bands(self):
        """
        Get bands required by this method.

        :return: required bands for left and right image
        :rtype: dict
        """

    @abstractmethod
    def run(
        self,
        left_image_object: xr.Dataset,
        right_image_object: xr.Dataset,
        saving_info_left=None,
        disp_lower_bound=None,
        disp_upper_bound=None,
        classif_bands_to_mask=None,
    ):
        """
        Run sparse matching on a pair of epipolar tiles.

        :param left_image_object: left epipolar tile
        :type left_image_object: xr.Dataset
        :param right_image_object: right epipolar tile
        :type right_image_object: xr.Dataset
        :param saving_info_left: saving information for outputs
        :type saving_info_left: dict
        :param disp_lower_bound: lower disparity bound
        :type disp_lower_bound: float or None
        :param disp_upper_bound: upper disparity bound
        :type disp_upper_bound: float or None
        :param classif_bands_to_mask: bands from classification to mask
        :type classif_bands_to_mask: list of str / int

        :return: left matches dataframe
        """

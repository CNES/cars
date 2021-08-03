#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
this module contains the otb geometry class
"""
import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import otbApplication
import rasterio as rio
import xarray as xr
from json_checker import And

from cars.conf import input_parameters
from cars.core import constants as cst
from cars.core import inputs
from cars.core.geometry import AbstractGeometry
from cars.core.otb_adapters import encode_to_otb


@AbstractGeometry.register_subclass("OTBGeometry")
class OTBGeometry(AbstractGeometry):
    """
    OTB geometry class
    """

    # TODO: remove the hard-coded import in the steps/__init__.py if this class
    # is removed from CARS

    @staticmethod
    def geo_conf_schema():
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

        geo_conf_schema = {
            input_parameters.IMG1_TAG: And(str, inputs.rasterio_can_open),
            input_parameters.IMG2_TAG: And(str, inputs.rasterio_can_open),
        }

        return geo_conf_schema

    @staticmethod
    def check_products_consistency(geo_conf: Dict[str, str]) -> bool:
        """
        Test if the product is readable by the OTB

        :param: the geometry configuration as requested by the geometry loader
        schema
        :return: True if the products are readable, False otherwise
        """
        # get images paths
        img1 = geo_conf[input_parameters.IMG1_TAG]
        img2 = geo_conf[input_parameters.IMG2_TAG]

        # test if both images have associated RPC
        status = False
        if OTBGeometry.check_geom_consistency(
            img1
        ) and OTBGeometry.check_geom_consistency(img2):
            status = True

        return status

    @staticmethod
    def check_geom_consistency(img: str) -> bool:
        """
        Check if the image have RPC information readable by the OTB

        :param img: path to the image
        :return: True if the RPC are readable, False otherwise
        """
        can_open_status = False
        try:
            geom_path = "./otb_can_open_test.geom"

            # try to dump .geom with ReadImageInfo app
            read_im_app = otbApplication.Registry.CreateApplication(
                "ReadImageInfo"
            )
            read_im_app.SetParameterString("in", img)
            read_im_app.SetParameterString("outkwl", geom_path)

            read_im_app.ExecuteAndWriteOutput()

            # check geom consistency
            if os.path.exists(geom_path):
                with open(geom_path) as geom_file_desc:
                    geom_dict = {}
                    for line in geom_file_desc:
                        key, val = line.split(": ")
                        geom_dict[key] = val
                    # pylint: disable=too-many-boolean-expressions
                    if (
                        "line_den_coeff_00" not in geom_dict
                        or "samp_den_coeff_00" not in geom_dict
                        or "line_num_coeff_00" not in geom_dict
                        or "samp_num_coeff_00" not in geom_dict
                        or "line_off" not in geom_dict
                        or "line_scale" not in geom_dict
                        or "samp_off" not in geom_dict
                        or "samp_scale" not in geom_dict
                        or "lat_off" not in geom_dict
                        or "lat_scale" not in geom_dict
                        or "long_off" not in geom_dict
                        or "long_scale" not in geom_dict
                        or "height_off" not in geom_dict
                        or "height_scale" not in geom_dict
                        or "polynomial_format" not in geom_dict
                    ):
                        logging.warning(
                            "No RPC model set for image {}".format(
                                geom_file_desc
                            )
                        )
                        can_open_status = False

                os.remove("./otb_can_open_test.geom")
                can_open_status = True
            else:
                logging.warning(
                    "{} does not have associated geom file".format(img)
                )
                can_open_status = False
        except Exception as read_error:
            logging.warning(
                "Exception caught while trying to read file {}: {}".format(
                    img, read_error
                )
            )
            can_open_status = False
        return can_open_status

    @staticmethod
    def triangulate(
        mode: str,
        data: xr.Dataset,
        grid1: str,
        grid2: str,
        geo_conf: Dict[str, str],
        min_elev1: float = None,
        max_elev1: float = None,
        min_elev2: float = None,
        max_elev2: float = None,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param mode: triangulation mode
        (cst.DISP_MODE or cst.MATCHES)
        :param data: cars disparity dataset
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param geo_conf: dictionary with the fields requested by the
        loader schema
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

        img1 = geo_conf[input_parameters.IMG1_TAG]
        img2 = geo_conf[input_parameters.IMG2_TAG]

        # Build triangulation app
        triangulation_app = otbApplication.Registry.CreateApplication(
            "EpipolarTriangulation"
        )

        if mode == cst.DISP_MODE:
            if roi_key is None:
                worker_logger = logging.getLogger("distributed.worker")
                worker_logger.warning(
                    "roi_key have to be set to use the "
                    "triangulation disparity mode"
                )
                raise Exception(
                    "roi_key have to be set to use the "
                    "triangulation disparity mode"
                )

            # encode disparity for otb
            disp = encode_to_otb(
                data[cst.DISP_MAP].values,
                data.attrs[cst.EPI_FULL_SIZE],
                data.attrs[roi_key],
            )

            # set disparity mode
            triangulation_app.SetParameterString("mode", "disp")
            triangulation_app.ImportImage("mode.disp.indisp", disp)
        elif mode == cst.MATCHES_MODE:
            # set matches mode
            triangulation_app.SetParameterString("mode", "sift")
            triangulation_app.SetImageFromNumpyArray(
                "mode.sift.inmatches", data
            )
        else:
            worker_logger = logging.getLogger("distributed.worker")
            worker_logger.warning(
                "Wrong triangulation mode (only 'disp' or 'matches'"
                " can be used as mode)"
            )
            raise Exception(
                "Wrong triangulation mode (only 'disp' or 'matches'"
                " can be used as mode)"
            )

        triangulation_app.SetParameterString("leftgrid", grid1)
        triangulation_app.SetParameterString("rightgrid", grid2)
        triangulation_app.SetParameterString("leftimage", img1)
        triangulation_app.SetParameterString("rightimage", img2)
        triangulation_app.SetParameterFloat("leftminelev", min_elev1)
        triangulation_app.SetParameterFloat("leftmaxelev", max_elev1)
        triangulation_app.SetParameterFloat("rightminelev", min_elev2)
        triangulation_app.SetParameterFloat("rightmaxelev", max_elev2)

        triangulation_app.Execute()

        llh = np.copy(triangulation_app.GetVectorImageAsNumpyArray("out"))

        return llh

    @staticmethod
    def generate_epipolar_grids(
        left_img: str,
        right_img: str,
        dem: Union[None, str] = None,
        default_alt: Union[None, float] = None,
        epipolar_step: int = 30,
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param left_img: path to left image
        :param right_img: path to right image
        :param dem: path to the dem folder
        :param default_alt: default altitude to use in the missing dem regions
        :param epipolar_step: step to use to construct the epipolar grids
        :return: Tuple composed of :
            - the left epipolar grid as a numpy array
            - the right epipolar grid as a numpy array
            - the left grid origin as a list of float
            - the left grid spacing as a list of float
            - the epipolar image size as a list of int
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float
        """
        stereo_app = otbApplication.Registry.CreateApplication(
            "StereoRectificationGridGenerator"
        )

        stereo_app.SetParameterString("io.inleft", left_img)
        stereo_app.SetParameterString("io.inright", right_img)
        stereo_app.SetParameterInt("epi.step", epipolar_step)
        if dem is not None:
            stereo_app.SetParameterString("epi.elevation.dem", dem)
        if default_alt is not None:
            stereo_app.SetParameterFloat("epi.elevation.default", default_alt)

        stereo_app.Execute()

        # Export grids to numpy
        left_grid_as_array = np.copy(
            stereo_app.GetVectorImageAsNumpyArray("io.outleft")
        )
        right_grid_as_array = np.copy(
            stereo_app.GetVectorImageAsNumpyArray("io.outright")
        )

        epipolar_size_x, epipolar_size_y, baseline = (
            stereo_app.GetParameterInt("epi.rectsizex"),
            stereo_app.GetParameterInt("epi.rectsizey"),
            stereo_app.GetParameterFloat("epi.baseline"),
        )

        origin = stereo_app.GetImageOrigin("io.outleft")
        spacing = stereo_app.GetImageSpacing("io.outleft")

        # Convert epipolar size depending on the pixel size
        # TODO: remove this patch when OTB issue
        # https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2176
        # is resolved
        with rio.open(left_img, "r") as rio_dst:
            pixel_size_x, pixel_size_y = (
                rio_dst.transform[0],
                rio_dst.transform[4],
            )

        mean_size = (abs(pixel_size_x) + abs(pixel_size_y)) / 2
        epipolar_size_x = int(np.floor(epipolar_size_x * mean_size))
        epipolar_size_y = int(np.floor(epipolar_size_y * mean_size))

        # we want disp_to_alt_ratio = resolution/(B/H), in m.pixel^-1
        disp_to_alt_ratio = 1 / baseline

        return (
            left_grid_as_array,
            right_grid_as_array,
            origin,
            spacing,
            [epipolar_size_x, epipolar_size_y],
            disp_to_alt_ratio,
        )

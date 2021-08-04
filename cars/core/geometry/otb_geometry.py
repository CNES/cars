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
from typing import List, Tuple, Union

import numpy as np
import otbApplication
import rasterio as rio
import xarray as xr

from cars.core import constants as cst
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
    def triangulate(
        mode: str,
        data: xr.Dataset,
        grid1: str,
        grid2: str,
        img1: str,
        img2: str,
        min_elev1: float,
        max_elev1: float,
        min_elev2: float,
        max_elev2: float,
        roi_key: str = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param mode: triangulation mode
        (cst.DISP_MODE or cst.MATCHES)
        :param data: cars disparity dataset
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param img1: path to image 1
        :param img2: path to image 2
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

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

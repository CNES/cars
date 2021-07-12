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
import numpy as np
import otbApplication
import xarray as xr

from cars.core import constants as cst
from cars.core.geometry import AbstractGeometry
from cars.core.otb_adapters import encode_to_otb


@AbstractGeometry.register_subclass("OTBGeometry")
class OTBGeometry(AbstractGeometry):
    """
    OTB geometry class
    """

    @staticmethod
    def triangulate(
        data: xr.Dataset,
        roi_key: str,
        grid1: str,
        grid2: str,
        img1: str,
        img2: str,
        min_elev1: float,
        max_elev1: float,
        min_elev2: float,
        max_elev2: float,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity dataset

        :param data: cars disparity dataset
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param img1: path to image 1
        :param img2: path to image 2
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :return: the long/lat/height numpy array in output of the triangulation
        """
        # encode disparity for otb
        disp = encode_to_otb(
            data[cst.DISP_MAP].values,
            data.attrs[cst.EPI_FULL_SIZE],
            data.attrs[roi_key],
        )

        # Build triangulation app
        triangulation_app = otbApplication.Registry.CreateApplication(
            "EpipolarTriangulation"
        )

        triangulation_app.SetParameterString("mode", "disp")
        triangulation_app.ImportImage("mode.disp.indisp", disp)

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
    def triangulate_matches(
        matches: np.ndarray,
        grid1: str,
        grid2: str,
        img1: str,
        img2: str,
        min_elev1: float,
        max_elev1: float,
        min_elev2: float,
        max_elev2: float,
    ) -> np.ndarray:
        """
        Performs triangulation from matches

        :param matches: input matches to triangulate
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param img1: path to image 1
        :param img2: path to image 2
        :param min_elev1: min elevation for image 1
        :param max_elev1: max elevation fro image 1
        :param min_elev2: min elevation for image 2
        :param max_elev2: max elevation for image 2
        :return: the long/lat/height numpy array in output of the triangulation
        """
        # Build triangulation app
        triangulation_app = otbApplication.Registry.CreateApplication(
            "EpipolarTriangulation"
        )

        triangulation_app.SetParameterString("mode", "sift")
        triangulation_app.SetImageFromNumpyArray("mode.sift.inmatches", matches)

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

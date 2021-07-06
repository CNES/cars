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
this module contains the otb triangulation class
"""
import numpy as np
import otbApplication
import xarray as xr

from cars.core import constants as cst
from cars.plugins.triangulation.abstract import AbstractTriangulation


@AbstractTriangulation.register_subclass("OTBTriangulation")
class OTBTriangulation(AbstractTriangulation):
    """
    OTB Triangulation class
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
        disp = OTBTriangulation.encode_to_otb(
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

    @staticmethod
    def encode_to_otb(data_array, largest_size, roi, origin=None, spacing=None):
        """
        This function encodes a numpy array with metadata
        so that it can be used by the ImportImage method of otb applications

        :param data_array: The numpy data array to encode
        :type data_array: numpy array
        :param largest_size: The size of the full image
            (data_array can be a part of a bigger image)
        :type largest_size: list of two int
        :param roi: Region encoded in data array ([x_min,y_min,x_max,y_max])
        :type roi: list of four int
        :param origin: Origin of full image (default origin: (0, 0))
        :type origin: list of two int
        :param spacing: Spacing of full image (default spacing: (1,1))
        :type spacing: list of two int
        :returns: A dictionary of attributes ready to be imported by ImportImage
        :rtype: dict
        """

        otb_origin = otbApplication.itkPoint()
        otb_origin[0] = origin[0] if origin is not None else 0
        otb_origin[1] = origin[1] if origin is not None else 0

        otb_spacing = otbApplication.itkVector()
        otb_spacing[0] = spacing[0] if spacing is not None else 1
        otb_spacing[1] = spacing[1] if spacing is not None else 1

        otb_largest_size = otbApplication.itkSize()
        otb_largest_size[0] = int(largest_size[0])
        otb_largest_size[1] = int(largest_size[1])

        otb_roi_region = otbApplication.itkRegion()
        otb_roi_region["size"][0] = int(roi[2] - roi[0])
        otb_roi_region["size"][1] = int(roi[3] - roi[1])
        otb_roi_region["index"][0] = int(roi[0])
        otb_roi_region["index"][1] = int(roi[1])

        output = {
            "origin": otb_origin,
            "spacing": otb_spacing,
            "size": otb_largest_size,
            "region": otb_roi_region,
            "metadata": otbApplication.itkMetaDataDictionary(),
            "array": data_array,
        }

        return output

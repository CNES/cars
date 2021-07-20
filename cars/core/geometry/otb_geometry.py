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

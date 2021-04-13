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
Preprocessing module:
contains functions used for triangulation
"""

import numpy as np
import xarray as xr
import otbApplication

from cars import constants as cst
from cars import utils
from cars.conf import input_parameters, \
                      output_compute_dsm, \
                      output_prepare


def triangulate_matches(configuration, matches, snap_to_img1=False):
    """
    This function will perform triangulation from sift matches

    :param configuration: StereoConfiguration
    :type configuration: StereoConfiguration
    :param matches: numpy.array of matches of shape (nb_matches, 4)
    :type data: numpy.ndarray
    :param snap_to_img1: If this is True, Lines of Sight of img2 are moved so
                         as to cross those of img1
    :param snap_to_img1: bool
    :returns: point_cloud as a dataset containing:

        * Array with shape (nb_matches,1,3), with last dimension
        corresponding to longitude, lattitude and elevation
        * Array with shape (nb_matches,1) with output mask

    :rtype: xarray.Dataset
    """

    # Retrieve information from configuration
    input_configuration = configuration[input_parameters.INPUT_SECTION_TAG]
    preprocessing_output_configuration = configuration\
        [output_prepare.PREPROCESSING_SECTION_TAG]\
        [output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG]

    img1 = input_configuration[input_parameters.IMG1_TAG]
    img2 = input_configuration[input_parameters.IMG2_TAG]

    grid1 = preprocessing_output_configuration[
        output_prepare.LEFT_EPIPOLAR_GRID_TAG]
    grid2 = preprocessing_output_configuration[
        output_prepare.RIGHT_EPIPOLAR_GRID_TAG]
    if snap_to_img1:
        grid2 = preprocessing_output_configuration\
            [output_compute_dsm.RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG]

    # Retrieve elevation range from imgs
    (min_elev1, max_elev1) = utils.get_elevation_range_from_metadata(img1)
    (min_elev2, max_elev2) = utils.get_elevation_range_from_metadata(img2)

    # Build triangulation app
    triangulation_app = otbApplication.Registry.CreateApplication(
        "EpipolarTriangulation")

    triangulation_app.SetParameterString("mode","sift")
    triangulation_app.SetImageFromNumpyArray("mode.sift.inmatches",matches)

    triangulation_app.SetParameterString("leftgrid", grid1)
    triangulation_app.SetParameterString("rightgrid", grid2)
    triangulation_app.SetParameterString("leftimage", img1)
    triangulation_app.SetParameterString("rightimage", img2)
    triangulation_app.SetParameterFloat("leftminelev",min_elev1)
    triangulation_app.SetParameterFloat("leftmaxelev",max_elev1)
    triangulation_app.SetParameterFloat("rightminelev",min_elev2)
    triangulation_app.SetParameterFloat("rightmaxelev",max_elev2)

    triangulation_app.Execute()

    llh = np.copy(triangulation_app.GetVectorImageAsNumpyArray("out"))

    row = np.array(range(llh.shape[0]))
    col = np.array([0])

    msk = np.full(llh.shape[0:2],255, dtype=np.uint8)

    point_cloud = xr.Dataset({cst.X: ([cst.ROW, cst.COL], llh[:, :, 0]),
                              cst.Y: ([cst.ROW, cst.COL], llh[:, :, 1]),
                              cst.Z: ([cst.ROW, cst.COL], llh[:, :, 2]),
                              cst.POINTS_CLOUD_CORR_MSK: ([cst.ROW, cst.COL],
                                                          msk)},
                             coords={cst.ROW: row,cst.COL: col})
    point_cloud.attrs[cst.EPSG] = int(4326)

    return point_cloud

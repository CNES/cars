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
Sift module:
contains sift sparse matching method
"""

# Standard imports
from __future__ import absolute_import

# Third party imports
import numpy as np
import otbApplication as otb

# CARS imports
from cars import otb_pipelines
from cars.core import constants as cst


def dataset_matching(
    ds1,
    ds2,
    matching_threshold=0.6,
    n_octave=8,
    n_scale_per_octave=3,
    dog_threshold=20,
    edge_threshold=5,
    magnification=2.0,
    backmatching=True,
):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param threshold: Threshold for matches
    :type threshold: float
    :param backmatching: Also check that right vs. left gives same match
    :type backmatching: bool
    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    size1 = [
        int(ds1.attrs["region"][2] - ds1.attrs["region"][0]),
        int(ds1.attrs["region"][3] - ds1.attrs["region"][1]),
    ]
    roi1 = [0, 0, size1[0], size1[1]]
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]

    size2 = [
        int(ds2.attrs["region"][2] - ds2.attrs["region"][0]),
        int(ds2.attrs["region"][3] - ds2.attrs["region"][1]),
    ]
    roi2 = [0, 0, size2[0], size2[1]]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]

    # Encode images for OTB
    im1 = otb_pipelines.encode_to_otb(
        ds1[cst.EPI_IMAGE].values, size1, roi1, origin=origin1
    )
    msk1 = otb_pipelines.encode_to_otb(
        ds1[cst.EPI_MSK].values, size1, roi1, origin=origin1
    )
    im2 = otb_pipelines.encode_to_otb(
        ds2[cst.EPI_IMAGE].values, size2, roi2, origin=origin2
    )
    msk2 = otb_pipelines.encode_to_otb(
        ds2[cst.EPI_MSK].values, size2, roi2, origin=origin2
    )

    # Build sift matching app
    matching_app = otb.Registry.CreateApplication("EpipolarSparseMatching")

    matching_app.ImportImage("in1", im1)
    matching_app.ImportImage("in2", im2)
    matching_app.EnableParameter("inmask1")
    matching_app.ImportImage("inmask1", msk1)
    matching_app.EnableParameter("inmask2")
    matching_app.ImportImage("inmask2", msk2)

    matching_app.SetParameterInt("maskvalue", 0)
    matching_app.SetParameterString("algorithm", "sift")
    matching_app.SetParameterFloat("matching", matching_threshold)
    matching_app.SetParameterInt("octaves", n_octave)
    matching_app.SetParameterInt("scales", n_scale_per_octave)
    matching_app.SetParameterFloat("tdog", dog_threshold)
    matching_app.SetParameterFloat("tedge", edge_threshold)
    matching_app.SetParameterFloat("magnification", magnification)
    matching_app.SetParameterInt("backmatching", backmatching)
    matching_app.Execute()

    # Retrieve number of matches
    nb_matches = matching_app.GetParameterInt("nbmatches")

    matches = np.empty((0, 4))

    if nb_matches > 0:
        # Export result to numpy
        matches = np.copy(
            matching_app.GetVectorImageAsNumpyArray("out")[:, :, -1]
        )

    return matches

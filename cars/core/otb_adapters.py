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
this module contains the data adapters to use the otb
"""
import otbApplication  # pylint: disable=import-error


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
    :return: A dictionary of attributes ready to be imported by ImportImage
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


def rigid_transform_resample(
    img: str, scale_x: float, scale_y: float, img_transformed: str
):
    """
    Execute RigidTransformResample OTB application

    :param img: path to the image to transform
    :param scale_x: scale factor to apply along x axis
    :param scale_y: scale factor to apply along y axis
    :param img_transformed: output image path
    """

    # create otb app to rescale input images
    app = otbApplication.Registry.CreateApplication("RigidTransformResample")

    app.SetParameterString("in", img)
    app.SetParameterString("transform.type", "id")
    app.SetParameterFloat("transform.type.id.scalex", abs(scale_x))
    app.SetParameterFloat("transform.type.id.scaley", abs(scale_y))
    app.SetParameterString("out", img_transformed)
    app.ExecuteAndWriteOutput()

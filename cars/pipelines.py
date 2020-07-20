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
This module contains functions that builds Orfeo ToolBox pipelines used by cars
"""

import numpy as np
import otbApplication


def build_stereorectification_grid_pipeline(img1, img2, dem=None, default_alt=None, epi_step=30):
    """
    This function builds the stereo-rectification pipeline and return it along with grids and sizes

    :param img1: Path to the left image
    :type img1: string
    :param img2: Path to right image
    :type img2: string
    :param dem: Path to DEM directory
    :type dem: string
    :param default_alt: Default altitude above geoid
    :type default_alt: float
    :param epi_step: Step of the stereo-rectification grid
    :type epi_step: int
    :returns: (left_grid, right_grid, epipolar_size_x, epipolar_size_y, pipeline) tuple
    :rtype: left grid and right_grid as otb::Image pointers, epipolar_size_xy  as int, baseline_ratio (resolution * B/H) as float, pipeline as a dictionnary containing applications composing the pipeline
    """
    stereo_app = otbApplication.Registry.CreateApplication(
        "StereoRectificationGridGenerator")
    stereo_app.SetParameterString("io.inleft", img1)
    stereo_app.SetParameterString("io.inright", img2)
    stereo_app.SetParameterInt("epi.step", epi_step)
    if dem is not None:
        stereo_app.SetParameterString("epi.elevation.dem", dem)
    if default_alt is not None:
        stereo_app.SetParameterFloat("epi.elevation.default", default_alt)
    stereo_app.Execute()

    pipeline = {"stereo_app": stereo_app}
    return stereo_app.GetParameterOutputImage("io.outleft"), \
        stereo_app.GetParameterOutputImage("io.outright"), \
        stereo_app.GetParameterInt("epi.rectsizex"), \
        stereo_app.GetParameterInt("epi.rectsizey"), \
        stereo_app.GetParameterFloat("epi.baseline"), pipeline


def build_extract_roi_application(img, region):
    """
    This function builds a ready to use instance of the ExtractROI application

    :param img: Pointer to the OTB image to extract
    :type img: otb::Image pointer
    :param region: Extraction region
    :type region: list of 4 int (xmin, ymin, xmax, ymax)
    :returns: (extracted image, roi application) tuple
    :rtype: extracted image as otb::Image pointer and ready to use instance of the ExtractROI application
    """
    extract_app = otbApplication.Registry.CreateApplication("ExtractROI")
    if isinstance(img, str):
        extract_app.SetParameterString("in", img)
    elif isinstance(img, np.ndarray):
        extract_app.SetImageFromNumpyArray("in", img)
    else:
        extract_app.SetParameterInputImage("in", img)
    extract_app.SetParameterInt("startx", int(region[0]))
    extract_app.SetParameterInt("starty", int(region[1]))
    extract_app.SetParameterInt("sizex", int(region[2]) - int(region[0]))
    extract_app.SetParameterInt("sizey", int(region[3]) - int(region[1]))
    extract_app.Execute()

    return extract_app.GetParameterOutputImage("out"), extract_app


def build_mask_pipeline(
        img,
        grid,
        nodata,
        mask,
        epipolar_size_x,
        epipolar_size_y,
        roi=None):
    """
    This function builds the a pipeline that computes and resampled image mask in epipolar geometry

    :param img: Path to the left image
    :type img: string
    :param grid: The stereo-rectification resampling grid
    :type grid: otb::Image pointer or string
    :param nodata: Pixel value to be treated as nodata in image or None
    :type nodata: float
    :param mask:  Path to left image mask or None
    :type mask: string
    :param epipolar_size_x: Size of stereo-rectified images in x
    :type epipolar_size_x: int
    :param epipolar_size_y: Size of stereo-rectified images in y
    :type epipolar_size_y: int
    :param roi: Region over which to compute epipolar mask or None
    :type roi: list of 4 int (xmin, ymin, xmax, ymax)
    :returns: (mask, pipeline) tuple
    :rtype: resampled mask as an otb::Image pointer and pipeline as a dictionnary containing applications composing the pipeline
    """
    mask_app = otbApplication.Registry.CreateApplication("BuildMask")
    mask_app.SetParameterString("in", img)
    if nodata is not None:
        mask_app.SetParameterFloat("innodata", nodata)
    if mask is not None:
        mask_app.SetParameterString("inmask", mask)
        mask_app.EnableParameter("inmask")
    mask_app.SetParameterFloat("outnodata", 255)
    mask_app.SetParameterFloat("outvalid", 0)
    mask_app.Execute()

    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling")
    resampling_app.SetParameterInputImage(
        "io.in", mask_app.GetParameterOutputImage("out"))

    if isinstance(grid, str):
        resampling_app.SetParameterString("grid.in", grid)
    else:
        resampling_app.SetParameterInputImage("grid.in", grid)

    resampling_app.SetParameterString("grid.type", "def")
    resampling_app.SetParameterInt("out.sizex", epipolar_size_x)
    resampling_app.SetParameterInt("out.sizey", epipolar_size_y)
    resampling_app.SetParameterString("interpolator", "nn")
    resampling_app.SetParameterFloat("out.default", 255)
    resampling_app.Execute()

    ret = resampling_app.GetParameterOutputImage("io.out")
    pipeline = {"mask_app": mask_app, "resampling_app": resampling_app}

    # TODO: Dilate nodata mask to ensure that interpolated pixels are not
    # contaminated
    if roi is not None:
        ret, extract_app = build_extract_roi_application(
            resampling_app.GetParameterOutputImage("io.out"), roi)
        pipeline["extract_app"] = extract_app

    return ret, pipeline


def build_bundletoperfectsensor_pipeline(
        pan,
        ms):
    """
    This function builds the a pipeline that performs P+XS pansharpening

    :param pan: Path to the panchromatic image
    :type pan: string
    :param ms: Path to the multispectral image
    :type ms: string
    :returns: (img, pipeline) tuple
    :rtype: pansharpened image as an otb::Image pointer and pipeline as a dictionnary containing applications composing the pipeline
    """
    pansharpening_app = otbApplication.Registry.CreateApplication(
        "BundleToPerfectSensor")
    pansharpening_app.SetParameterString("inp", pan)
    pansharpening_app.SetParameterString("inxs", ms)
    pansharpening_app.Execute()

    pipeline = {"pansharpening_app": pansharpening_app}
    ret = pansharpening_app.GetParameterOutputImage("out")

    return ret, pipeline


def build_image_resampling_pipeline(
        img,
        grid,
        epipolar_size_x,
        epipolar_size_y,
        roi=None):
    """
    This function builds the a pipeline that resamples images in epipolar geometry

    :param img: Path to the left image
    :type img: string
    :param grid: The stereo-rectification resampling grid
    :type grid: otb::Image pointer or string
    :param epipolar_size_x: Size of stereo-rectified images in x
    :type epipolar_size_x: int
    :param epipolar_size_y: Size of stereo-rectified images in y
    :type epipolar_size_y: int
    :param roi: Region over which to compute epipolar images, or None
    :type roi: list of 4 int (xmin, ymin, xmax, ymax)
    :returns: (img, pipeline) tuple
    :rtype: resampled image as an otb::Image pointer and pipeline as a dictionnary containing applications composing the pipeline
    """
    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling")
    if isinstance(img, str):
        resampling_app.SetParameterString("io.in", img)
    else:
        resampling_app.SetParameterInputImage("io.in", img)

    if isinstance(grid, str):
        resampling_app.SetParameterString("grid.in", grid)
    else:
        resampling_app.SetParameterInputImage("grid.in", grid)

    resampling_app.SetParameterString("grid.type", "def")
    resampling_app.SetParameterInt("out.sizex", epipolar_size_x)
    resampling_app.SetParameterInt("out.sizey", epipolar_size_y)

    resampling_app.Execute()

    pipeline = {"resampling_app": resampling_app}
    ret = resampling_app.GetParameterOutputImage("io.out")

    if roi is not None:
        ret, extract_app = build_extract_roi_application(
            resampling_app.GetParameterOutputImage("io.out"), roi)
        pipeline["extract_app"] = extract_app

    return ret, pipeline


def encode_to_otb(
    data_array, largest_size, roi, origin=[
        0, 0], spacing=[
            1, 1]):
    """
    This function encodes a numpy array with metadata so that it can be used by the ImportImage method of otb applications

    :param data_array: The numpy data array to encode
    :type data_array: numpy array
    :param largest_size: The size of the full image (data_array can be a part of a bigger image)
    :type largest_size: list of two int
    :param roi: Region encoded in data array ([x_min,y_min,x_max,y_max])
    :type roi: list of four int
    :param origin: Origin of full image
    :type origin: list of two int
    :param spacing: Spacing of full image
    :type spacing: list of two int
    :returns: A dictionnary of attributes ready to be imported by ImportImage
    :rtype: dict
    """

    otb_origin = otbApplication.itkPoint()
    otb_origin[0] = origin[0]
    otb_origin[1] = origin[1]

    otb_spacing = otbApplication.itkVector()
    otb_spacing[0] = spacing[0]
    otb_spacing[1] = spacing[1]

    otb_largest_size = otbApplication.itkSize()
    otb_largest_size[0] = int(largest_size[0])
    otb_largest_size[1] = int(largest_size[1])

    otb_roi_region = otbApplication.itkRegion()
    otb_roi_region['size'][0] = int(roi[2] - roi[0])
    otb_roi_region['size'][1] = int(roi[3] - roi[1])
    otb_roi_region['index'][0] = int(roi[0])
    otb_roi_region['index'][1] = int(roi[1])

    output = {}

    output['origin'] = otb_origin
    output['spacing'] = otb_spacing
    output['size'] = otb_largest_size
    output['region'] = otb_roi_region
    output['metadata'] = otbApplication.itkMetaDataDictionary()
    output['array'] = data_array

    return output

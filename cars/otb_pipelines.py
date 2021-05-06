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
Pipelines module:
contains functions that builds Orfeo ToolBox pipelines used by cars
"""

from __future__ import absolute_import
import numpy as np
import otbApplication

from cars.conf import mask_classes


def build_stereorectification_grid_pipeline(
        img1,
        img2,
        dem = None,
        default_alt = None,
        epi_step = 30):
    """
    This function builds the stereo-rectification pipeline and
    return it along with grids and sizes

    :param img1: Path to the left image
    :type img1: string
    :param img2: Path to right image
    :type img2: string
    :param dem: Path to DEM directory
    :type dem: string
    :param default_alt: Default altitude above ellipsoid
    :type default_alt: float
    :param epi_step: Step of the stereo-rectification grid
    :type epi_step: int
    :returns: (left_grid, right_grid,
            epipolar_size_x, epipolar_size_y, pipeline) tuple
    :rtype: left grid and right_grid as numpy arrays,
            origin as two float,
            spacing as two float,
            epipolar_size_xy  as int,
            baseline_ratio (resolution * B/H) as float,
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

    # Export grids to numpy
    left_grid_as_array = np.copy(
        stereo_app.GetVectorImageAsNumpyArray("io.outleft"))
    right_grid_as_array = np.copy(
        stereo_app.GetVectorImageAsNumpyArray("io.outright"))

    epipolar_size_x, epipolar_size_y, baseline = \
        stereo_app.GetParameterInt("epi.rectsizex"), \
        stereo_app.GetParameterInt("epi.rectsizey"), \
        stereo_app.GetParameterFloat("epi.baseline")

    origin = stereo_app.GetImageOrigin(
        "io.outleft")
    spacing = stereo_app.GetImageSpacing(
        "io.outleft")


    return left_grid_as_array, right_grid_as_array, \
        origin, spacing, epipolar_size_x, epipolar_size_y, baseline


def build_extract_roi_application(img, region):
    """
    This function builds a ready to use instance of the ExtractROI application

    :param img: Pointer to the OTB image to extract
    :type img: otb::Image pointer
    :param region: Extraction region
    :type region: list of 4 int (xmin, ymin, xmax, ymax)
    :returns: (extracted image, roi application) tuple
    :rtype: ready to use instance of the ExtractROI application
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

    return extract_app


def build_mask_pipeline(
        img,
        grid,
        nodata,
        mask,
        epipolar_size_x,
        epipolar_size_y,
        roi):
    """
    This function builds a pipeline that computes and
    resampled image mask in epipolar geometry

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
    :returns: mask
    :rtype: resampled mask as numpy array
    """
    mask_app = otbApplication.Registry.CreateApplication("BuildMask")

    mask_app.SetParameterString("in", img)
    if nodata is not None:
        mask_app.SetParameterFloat("innodata", nodata)
    if mask is not None:
        mask_app.SetParameterString("inmask", mask)
        mask_app.EnableParameter("inmask")
    mask_app.SetParameterFloat("outnodata",
        mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION)
    mask_app.SetParameterFloat("outvalid", mask_classes.VALID_VALUE)

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
    resampling_app.SetParameterFloat("out.default",
        mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION)
    resampling_app.Execute()

    # TODO: Dilate nodata mask to ensure that interpolated pixels are not
    # contaminated
    extract_app = build_extract_roi_application(
        resampling_app.GetParameterOutputImage("io.out"), roi)
    msk = np.copy(extract_app.GetImageAsNumpyArray("out"))

    return msk

def build_bundletoperfectsensor_pipeline(
        pan_img,
        ms_img):
    """
    This function builds the a pipeline that performs P+XS pansharpening

    :param pan_img: Path to the panchromatic image
    :type pan_img: string
    :param ms_img: Path to the multispectral image
    :type ms_img: string
    :returns: resample_image
    :rtype: otb application
    """
    pansharpening_app = otbApplication.Registry.CreateApplication(
        "BundleToPerfectSensor")

    pansharpening_app.SetParameterString("inp", pan_img)
    pansharpening_app.SetParameterString("inxs", ms_img)

    pansharpening_app.Execute()

    return pansharpening_app


def build_image_resampling_pipeline(
        img,
        grid,
        epipolar_size_x,
        epipolar_size_y,
        roi,
        lowres_color=None):
    """
    This function builds a pipeline that resamples images in epipolar geometry

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
    :param lowres_color: Path to the low resolution color image
    ;type lowres_color: string
    :returns: resampled image
    :rtype: resampled image as numpy array
    """

    # Build bundletoperfectsensor (p+xs fusion) for images
    if lowres_color is not None:
        pansharpening_app = build_bundletoperfectsensor_pipeline(
            img, lowres_color)
        img = pansharpening_app.GetParameterOutputImage("out")

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

    extract_app = build_extract_roi_application(
        resampling_app.GetParameterOutputImage("io.out"), roi)

    # Retrieve data and build left dataset
    resampled = np.copy(extract_app.GetVectorImageAsNumpyArray("out"))

    return resampled


def encode_to_otb(
    data_array, largest_size, roi, origin=None, spacing=None):
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

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
OTB Pipelines module:
contains functions that builds Orfeo ToolBox pipelines used by CARS
Refacto: Split function in generic externals calls through functional steps
         interfaces (epipolar rectification, ...)
"""

# Standard imports
from __future__ import absolute_import

from typing import List

# Third party imports
import numpy as np
import otbApplication
import rasterio as rio
import xarray as xr

# CARS imports
from cars.conf import mask_classes
from cars.core import constants as cst


def build_stereorectification_grid_pipeline(
    img1, img2, dem=None, default_alt=None, epi_step=30
):
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
        "StereoRectificationGridGenerator"
    )

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
    with rio.open(img1, "r") as rio_dst:
        pixel_size_x, pixel_size_y = rio_dst.transform[0], rio_dst.transform[4]

    mean_size = (abs(pixel_size_x) + abs(pixel_size_y)) / 2
    epipolar_size_x = int(np.floor(epipolar_size_x * mean_size))
    epipolar_size_y = int(np.floor(epipolar_size_y * mean_size))

    return (
        left_grid_as_array,
        right_grid_as_array,
        origin,
        spacing,
        epipolar_size_x,
        epipolar_size_y,
        baseline,
    )


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
    img, grid, nodata, mask, epipolar_size_x, epipolar_size_y, roi
):
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
    mask_app.SetParameterFloat(
        "outnodata", mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION
    )
    mask_app.SetParameterFloat("outvalid", mask_classes.VALID_VALUE)

    mask_app.Execute()

    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling"
    )
    resampling_app.SetParameterInputImage(
        "io.in", mask_app.GetParameterOutputImage("out")
    )

    if isinstance(grid, str):
        resampling_app.SetParameterString("grid.in", grid)
    else:
        resampling_app.SetParameterInputImage("grid.in", grid)

    resampling_app.SetParameterString("grid.type", "def")
    resampling_app.SetParameterInt("out.sizex", epipolar_size_x)
    resampling_app.SetParameterInt("out.sizey", epipolar_size_y)
    resampling_app.SetParameterString("interpolator", "nn")
    resampling_app.SetParameterFloat(
        "out.default", mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION
    )
    resampling_app.Execute()

    # TODO: Dilate nodata mask to ensure that interpolated pixels are not
    # contaminated
    extract_app = build_extract_roi_application(
        resampling_app.GetParameterOutputImage("io.out"), roi
    )
    msk = np.copy(extract_app.GetImageAsNumpyArray("out"))

    return msk


def build_bundletoperfectsensor_pipeline(pan_img, ms_img):
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
        "BundleToPerfectSensor"
    )

    pansharpening_app.SetParameterString("inp", pan_img)
    pansharpening_app.SetParameterString("inxs", ms_img)

    pansharpening_app.Execute()

    return pansharpening_app


def build_image_resampling_pipeline(
    img, grid, epipolar_size_x, epipolar_size_y, roi, lowres_color=None
):
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
            img, lowres_color
        )
        img = pansharpening_app.GetParameterOutputImage("out")

    resampling_app = otbApplication.Registry.CreateApplication(
        "GridBasedImageResampling"
    )

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
        resampling_app.GetParameterOutputImage("io.out"), roi
    )

    # Retrieve data and build left dataset
    resampled = np.copy(extract_app.GetVectorImageAsNumpyArray("out"))

    return resampled


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

    output = {}

    output["origin"] = otb_origin
    output["spacing"] = otb_spacing
    output["size"] = otb_largest_size
    output["region"] = otb_roi_region
    output["metadata"] = otbApplication.itkMetaDataDictionary()
    output["array"] = data_array

    return output


def image_envelope(img, shp, dem=None, default_alt=None):
    """
    Export the image footprint to a shapefile
    TODO: refacto with externals (OTB) and steps.

    :param img: filename to image or OTB pointer to image
    :type img:  string or OTBImagePointer
    :param shp: Path to the output shapefile
    :type shp: string
    :param dem: Directory containing DEM tiles
    :type dem: string
    :param default_alt: Default altitude above ellipsoid
    :type default_alt: float
    """

    app = otbApplication.Registry.CreateApplication("ImageEnvelope")

    if isinstance(img, str):
        app.SetParameterString("in", img)
    else:
        app.SetParameterInputImage("in", img)

    if dem is not None:
        app.SetParameterString("elev.dem", dem)

    if default_alt is not None:
        app.SetParameterFloat("elev.default", default_alt)

    app.SetParameterString("out", shp)
    app.ExecuteAndWriteOutput()


def sensor_to_geo(
    img: str,
    x_coord: float,
    y_coord: float,
    z_coord: float = None,
    dem: str = None,
    geoid: str = None,
    default_elevation: float = None,
) -> np.ndarray:
    """
    For a given image point, compute the latitude, longitude, altitude

    Be careful: When SRTM is used, the default elevation (altitude)
    doesn't work anymore (OTB function) when ConvertSensorToGeoPointFast
    is called again. Check the values.

    Advice: to be sure, use x,y,z inputs only

    :param img: Path to an image
    :param x_coord: X Coordinate in input image sensor
    :param y_coord: Y Coordinate in input image sensor
    :param z_coord: Z Altitude coordinate to take the image
    :param dem: if z not defined, take this DEM directory input
    :param geoid: if z and dem not defined, take GEOID directory input
    :param elevation: if z, dem, geoid not defined, take default elevation
    :return: Latitude, Longitude, Altitude coordinates as a numpy array
    """
    s2c_app = otbApplication.Registry.CreateApplication(
        "ConvertSensorToGeoPointFast"
    )

    s2c_app.SetParameterString("in", img)
    s2c_app.SetParameterFloat("input.idx", x_coord)
    s2c_app.SetParameterFloat("input.idy", y_coord)

    if z_coord is not None:
        s2c_app.SetParameterFloat("input.idz", z_coord)
    elif dem is not None:
        s2c_app.SetParameterString("elevation.dem", dem)
    elif geoid is not None:
        s2c_app.SetParameterString("elevation.geoid", geoid)
    elif default_elevation is not None:
        s2c_app.SetParameterFloat("elevation.default", default_elevation)
    # else ConvertSensorToGeoPointFast have only X, Y and OTB configured GEOID

    s2c_app.Execute()

    lon = s2c_app.GetParameterFloat("output.idx")
    lat = s2c_app.GetParameterFloat("output.idy")
    alt = s2c_app.GetParameterFloat("output.idz")

    return np.array([lat, lon, alt])


def get_utm_zone_as_epsg_code(lon, lat):
    """
    Returns the EPSG code of the UTM zone where the lat, lon point falls in
    TODO: refacto with externals (OTB)

    :param lon: longitude of the point
    :type lon: float
    :param lat: lattitude of the point
    :type lat: float
    :returns: The EPSG code corresponding to the UTM zone
    :rtype: int
    """
    utm_app = otbApplication.Registry.CreateApplication(
        "ObtainUTMZoneFromGeoPoint"
    )
    utm_app.SetParameterFloat("lon", float(lon))
    utm_app.SetParameterFloat("lat", float(lat))
    utm_app.Execute()
    zone = utm_app.GetParameterInt("utm")
    north_south = 600 if lat >= 0 else 700
    return 32000 + north_south + zone


def read_lowres_dem(
    startx,
    starty,
    sizex,
    sizey,
    dem=None,
    default_alt=None,
    resolution=0.000277777777778,
):
    """
    Read an extract of the low resolution input DSM and return it as an Array

    :param startx: Upper left x coordinate for grid in WGS84
    :type startx: float
    :param starty: Upper left y coordinate for grid in WGS84
        (remember that values are decreasing in y axis)
    :type starty: float
    :param sizex: Size of grid in x direction
    :type sizex: int
    :param sizey: Size of grid in y direction
    :type sizey: int
    :param dem: DEM directory
    :type dem: string
    :param default_alt: Default altitude above ellipsoid
    :type default_alt: float
    :param resolution: Resolution (in degrees) of output raster
    :type resolution: float
    :return: The extract of the lowres DEM as an xarray.Dataset
    :rtype: xarray.Dataset
    """

    app = otbApplication.Registry.CreateApplication("DEMReader")

    if dem is not None:
        app.SetParameterString("elev.dem", dem)

    if default_alt is not None:
        app.SetParameterFloat("elev.default", default_alt)

    app.SetParameterFloat("originx", startx)
    app.SetParameterFloat("originy", starty)
    app.SetParameterInt("sizex", sizex)
    app.SetParameterInt("sizey", sizey)
    app.SetParameterFloat("resolution", resolution)
    app.Execute()

    dem_as_array = np.copy(app.GetImageAsNumpyArray("out"))

    x_values_1d = np.linspace(
        startx + 0.5 * resolution,
        startx + resolution * (sizex + 0.5),
        sizex,
        endpoint=False,
    )
    y_values_1d = np.linspace(
        starty - 0.5 * resolution,
        starty - resolution * (sizey + 0.5),
        sizey,
        endpoint=False,
    )

    dims = [cst.Y, cst.X]
    coords = {cst.X: x_values_1d, cst.Y: y_values_1d}
    dsm_as_ds = xr.Dataset(
        {cst.RASTER_HGT: (dims, dem_as_array)}, coords=coords
    )
    dsm_as_ds[cst.EPSG] = 4326
    dsm_as_ds[cst.RESOLUTION] = resolution

    return dsm_as_ds


def read_image(raster_path: str, out_kwl_path: str):
    """
    use ReadImageInfo otb application with the .geom file dump

    :param raster_path: path to the image
    :param out_kwl_path: path to the output .geom file
    """
    read_im_app = otbApplication.Registry.CreateApplication("ReadImageInfo")
    read_im_app.SetParameterString("in", raster_path)
    read_im_app.SetParameterString("outkwl", out_kwl_path)

    read_im_app.ExecuteAndWriteOutput()


def triangulation_matches(
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


def triangulation(
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
    :param roi_key: dataset roi to use (can be cst.ROI or cst.ROI_WITH_MARGINS)
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

    # Build triangulation application
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


def epipolar_sparse_matching(
    ds1: xr.Dataset,
    roi1: List[int],
    size1: List[int],
    origin1: List[float],
    ds2: xr.Dataset,
    roi2: List[int],
    size2: List[int],
    origin2: List[float],
    matching_threshold: float,
    n_octave: int,
    n_scale_per_octave: int,
    dog_threshold: int,
    edge_threshold: int,
    magnification: float,
    backmatching: bool,
) -> np.ndarray:
    """
    Compute SIFT using vlfeat and performs epipolar sparse matching

    :param ds1: epipolar image 1 dataset
    :param roi1: roi to use for image 1
    :param size1: full epipolar image 1 size
    :param origin1: origin of full epipolar image 1
    :param ds2:  epipolar image 2 dataset
    :param roi2: roi to use for image 2
    :param size2: full epipolar image 2 size
    :param origin2: origin of full epipolar image 2
    :param matching_threshold: matching threshold to use in vlfeat
    :param n_octave: number of octaves to use in vlfeat
    :param n_scale_per_octave: number of scales per octave to use in vlfeat
    :param dog_threshold: difference of gaussians threshold to use in vlfeat
    :param edge_threshold: edge threshold to use in vlfeat
    :param magnification: magnification value to use in vlfeat
    :param backmatching: activation status of the back matching in vlfeat
    :return: matches as numpy array
    """
    # Encode images for OTB
    im1 = encode_to_otb(ds1[cst.EPI_IMAGE].values, size1, roi1, origin=origin1)
    msk1 = encode_to_otb(ds1[cst.EPI_MSK].values, size1, roi1, origin=origin1)
    im2 = encode_to_otb(ds2[cst.EPI_IMAGE].values, size2, roi2, origin=origin2)
    msk2 = encode_to_otb(ds2[cst.EPI_MSK].values, size2, roi2, origin=origin2)

    # create OTB matching application
    matching_app = otbApplication.Registry.CreateApplication(
        "EpipolarSparseMatching"
    )

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


def rigid_transform_resample(
    img: str, scalex: float, scaley: float, img_transformed: str
):
    """
    Execute RigidTransformResample OTB application

    :param img: path to the image to transform
    :param scalex: scale factor to apply along x axis
    :param scaley: scale factor to apply along y axis
    :param img_transform: output image path
    """

    # create otb app to rescale input images
    app = otbApplication.Registry.CreateApplication("RigidTransformResample")

    app.SetParameterString("in", img)
    app.SetParameterString("transform.type", "id")
    app.SetParameterFloat("transform.type.id.scalex", abs(scalex))
    app.SetParameterFloat("transform.type.id.scaley", abs(scaley))
    app.SetParameterString("out", img_transformed)
    app.ExecuteAndWriteOutput()

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
this module contains tools for the dem generation
"""
import contextlib
import logging
import os

import numpy as np
import pyproj
import rasterio as rio
import xdem

# Third-party imports
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import median_filter

from cars.core import preprocessing


def fit_initial_elevation_on_dem_median(
    dem_to_fit_path: str, dem_ref_path: str, dem_out_path: str
):
    """
    Coregistrates the two DEMs given then saves the result.
    The initial elevation will be cropped to reduce computation costs.
    Returns the transformation applied.

    :param dem_to_fit_path: Path to the dem to be fitted
    :type dem_to_fit_path: str
    :param dem_ref_path: Path to the dem to fit onto
    :type dem_ref_path: str
    :param dem_out_path: Path to save the resulting dem into
    :type dem_out_path: str

    :return: coregistration transformation applied
    :rtype: dict
    """
    # suppress all outputs of xdem
    with open(os.devnull, "w", encoding="utf8") as devnull:
        with (
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):

            # load DEMs
            dem_to_fit = xdem.DEM(dem_to_fit_path)
            dem_ref = xdem.DEM(dem_ref_path)

            # get the crs needed to reproject the data
            crs_out = dem_ref.crs
            crs_metric = dem_ref.get_metric_crs()

            # Crop dem_to_fit with dem_ref to reduce
            # computation costs.
            bbox = dem_ref.bounds
            bbox = add_margin(bbox)
            dem_to_fit = dem_to_fit.crop(bbox).reproject(crs=crs_metric)
            # Reproject dem_ref to dem_to_fit resolution to reduce
            # computation costs
            dem_ref = dem_ref.reproject(dem_to_fit)
            bbox = dem_ref.bounds
            bbox = add_margin(bbox)

            coreg_pipeline = xdem.coreg.NuthKaab()

            try:
                # fit dem_to_fit onto dem_ref, crop it, then reproject it
                # set a random state to always get the same results
                fit_dem = (
                    coreg_pipeline.fit_and_apply(
                        dem_ref, dem_to_fit, random_state=0
                    )
                    .crop(bbox)
                    .reproject(crs=crs_out)
                )
                # save the results
                fit_dem.save(dem_out_path)
                coreg_offsets = coreg_pipeline.meta["outputs"]["affine"]
            except (ValueError, AssertionError, TypeError):
                logging.warning(
                    "xDEM coregistration failed. This can happen when sensor "
                    "images are too small. No shift will be applied on DEM"
                )
                coreg_offsets = None

    return coreg_offsets


def add_margin(bbox, ratio=1):
    """
    Add margin to a bounding box
    :param bbox: input bounding box
    :type bbox: rasterio.coords.BoundingBox
    :param ratio: factor of bbox size to add to each side of bbox
    :type ratio: float

    :return: bounding box with margins
    :rtype: rasterio.coords.BoundingBox
    """
    try:
        assert bbox.left < bbox.right
        assert bbox.bottom < bbox.top
        width = bbox.right - bbox.left
        height = bbox.top - bbox.bottom
        new_left = bbox.left - ratio * width
        new_right = bbox.right + ratio * width
        new_bottom = bbox.bottom - ratio * height
        new_top = bbox.top + ratio * height
        new_bbox = BoundingBox(new_left, new_bottom, new_right, new_top)
    except AssertionError:
        logging.warning("Bounding box {} cannot be read".format(bbox))
        new_bbox = bbox
    return new_bbox


def generate_grid(
    pd_pc, resolution, xmin=None, xmax=None, ymin=None, ymax=None
):
    """
    Generate regular grid

    :param pd_pc: point cloud
    :type pd_pc: Pandas Dataframe
    :param resolution: resolution in meter
    :type resolution: float
    :param xmin: x min position in metric system
    :type xmin: float
    :param xmax: x max position in metric system
    :type xmax: float
    :param ymin: y min position in metric system
    :type ymin: float
    :param ymax: y max position in metric system
    :type ymax: float

    :return: regular grid
    :rtype: numpy array

    """

    if None in (xmin, xmax, ymin, ymax):
        mins = pd_pc.min(skipna=True)
        maxs = pd_pc.max(skipna=True)
        xmin = mins["x"]
        ymin = mins["y"]
        xmax = maxs["x"]
        ymax = maxs["y"]

    nb_x = int((xmax - xmin) / resolution)
    x_range = np.linspace(xmin, xmax, nb_x)
    nb_y = int((ymax - ymin) / resolution)

    y_range = np.linspace(ymin, ymax, nb_y)
    x_grid, y_grid = np.meshgrid(x_range, y_range)  # 2D grid for interpolation

    return x_grid, y_grid


def compute_stats(diff):
    """
    Compute and display statistics of difference between two DEM :
    Minimum, median, percentiles and maximum

    :param diff: altimetric difference between two DEM
    :type diff: numpy.array

    """
    mini = ("Min", np.nanmin(diff))
    median = ("Median", np.nanmedian(diff))
    p90 = ("p90", np.nanpercentile(diff, 90))
    p95 = ("p95", np.nanpercentile(diff, 95))
    p99 = ("p99", np.nanpercentile(diff, 99))
    maxi = ("Max", np.nanmax(diff))
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"| {mini[0]:6} | {median[0]:6} | {p90[0]:6} | "
        f"{p95[0]:6} | {p99[0]:6} | {maxi[0]:6} |"
    )
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"| {mini[1]:6.2f} | {median[1]:6.2f} | {p90[1]:6.2f} | "
        f"{p95[1]:6.2f} | {p99[1]:6.2f} | {maxi[1]:6.2f} |"
    )


def edit_transform(input_dem, resolution=None, transform=None):
    """
    Change transform of an image
    :param input_res: path of image
    :type input_dem: str
    :param resolution: image resolution for new transform to apply
    :type resolution: float
    :param transform: new transform to apply if resolution is not given
    :type transform: affine.Affine
    """
    if resolution is not None:
        if transform is None:
            transform = Affine.from_gdal(0, resolution, 0, 0, 0, resolution)
        else:
            raise ValueError(
                "Function edit_transform take resolution or "
                "transform as parameter but not both"
            )
    with rio.open(input_dem, "r+") as in_dem:
        previous_transform = in_dem.transform
        in_dem.transform = transform
    return previous_transform


def reverse_dem(input_dem):
    """
    Compute the opposite of a DEM :
    Altitudes sign is changed

    :param input_dem: path of DEM to reverse
    :type input_dem: str
    """
    with rio.open(input_dem, "r") as in_dem:
        data = in_dem.read()
        metadata = in_dem.meta
        nodata = in_dem.nodata
    with rio.open(input_dem, "w", **metadata) as out_dem:
        out_dem.write(-data)
        out_dem.nodata = -nodata


def downsample_dem(
    input_dem,
    scale,
    median_filter_size=7,
    default_alt=0,
):
    """
    Downsample median DEM with median resampling

    :param input_dem: path of DEM to downsample (only one band)
    :type input_dem: str
    """
    with rio.open(input_dem) as in_dem:
        data = in_dem.read(1)
        metadata = in_dem.meta
        src_transform = in_dem.transform
        width = in_dem.width
        height = in_dem.height
        crs = in_dem.crs
        nodata = in_dem.nodata

    dst_transform = src_transform * Affine.scale(scale)
    dst_height = int(height // scale) + 1
    dst_width = int(width // scale) + 1
    metadata["transform"] = dst_transform
    metadata["height"] = dst_height
    metadata["width"] = dst_width
    dem_data = np.zeros((dst_height, dst_width))
    reproject(
        data,
        dem_data,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=nodata,
        resampling=Resampling.med,
    )

    # Post-processing

    # Median filter
    dem_data = median_filter(dem_data, size=median_filter_size)

    # Fill nodata
    dem_data = rio.fill.fillnodata(
        dem_data,
        mask=~(dem_data == nodata),
    )

    dem_data[dem_data == nodata] = default_alt

    with rio.open(input_dem, "w", **metadata) as dst:
        dst.write(dem_data, 1)


def modify_terrain_bounds(
    bounds_poly, in_epsg, out_epsg, constant_margin, linear_margin=0
):
    """
    Modify the terrain bounds

    :param bounds_poly: Input region of interest for DEM
    :type bounds_poly: list
    :param in_epsg: EPSG code of dem_roi_to_use
    :type in_epsg: int
    :param out_epsg: EPSG code of dem_roi_to_use
    :type out_epsg: int
    :param margin: Margin of the output ROI in meters
    :type margin: int
    """
    # Get bounds
    xmin = min(bounds_poly[0], bounds_poly[2])
    xmax = max(bounds_poly[0], bounds_poly[2])
    ymin = min(bounds_poly[1], bounds_poly[3])
    ymax = max(bounds_poly[1], bounds_poly[3])

    bounds_cloud = [xmin, ymin, xmax, ymax]

    if in_epsg == 4326:
        # Convert resolution and margin to degrees
        utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
        conversion_factor = preprocessing.get_conversion_factor(
            bounds_cloud, 4326, utm_epsg
        )
        constant_margin *= conversion_factor

    # Get borders, adding margin

    xmin = xmin - constant_margin - linear_margin * (xmax - xmin)
    ymin = ymin - constant_margin - linear_margin * (ymax - ymin)
    xmax = xmax + constant_margin + linear_margin * (xmax - xmin)
    ymax = ymax + constant_margin + linear_margin * (ymax - ymin)

    terrain_bounds = [xmin, ymin, xmax, ymax]

    if out_epsg != in_epsg:
        crs_in = pyproj.CRS.from_epsg(in_epsg)
        crs_out = pyproj.CRS.from_epsg(out_epsg)

        transformer = pyproj.Transformer.from_crs(
            crs_in, crs_out, always_xy=True
        )

        xymin = transformer.transform(terrain_bounds[0], terrain_bounds[1])
        xymax = transformer.transform(terrain_bounds[2], terrain_bounds[3])

        xmin, ymin = xymin if isinstance(xymin, tuple) else (None, None)
        xmax, ymax = xymax if isinstance(xymax, tuple) else (None, None)

        if None in (xmin, ymin, xmax, ymax):
            raise RuntimeError("An error occured during the projection")

    new_terrain_bounds = [xmin, ymin, xmax, ymax]

    return new_terrain_bounds


def reproject_dem(dsm_file_name, epsg_out, out_file_name):
    """
    Reproject the DEM

    :param dsm_file_name: the path to dsm
    :type dsm_file_name: str
    :param epsg_out: the epsg code
    :type epsg_out: int
    :param out_file_name: the out path file
    :type out_file_name: str
    """
    with rio.open(dsm_file_name) as src:
        transform, width, height = calculate_default_transform(
            src.crs, epsg_out, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": epsg_out,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rio.open(out_file_name, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=epsg_out,
                    resampling=Resampling.nearest,
                )

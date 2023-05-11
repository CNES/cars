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
Outputs module:
contains some CARS global shared general purpose output functions
"""
# Standard imports
from typing import Union

# Third party imports
import fiona
import numpy as np
import pandas
import rasterio as rio
import xarray as xr
from fiona.crs import from_epsg  # pylint: disable=no-name-in-module
from rasterio.profiles import DefaultGTiffProfile
from shapely.geometry import mapping

# CARS imports
from cars.core import constants as cst


def write_ply(path_ply_file: str, cloud: Union[xr.Dataset, pandas.DataFrame]):
    """
    Write cloud to a ply file

    :param path: path to the ply file to write
    :param cloud: cloud to write,
        it can be a xr.Dataset as the ones given in output of the triangulation
        or a pandas.DataFrame as used in the rasterization
    """

    with open(path_ply_file, "w", encoding="utf-8") as ply_file:
        if isinstance(cloud, xr.Dataset):
            nb_points = int(
                cloud[cst.POINTS_CLOUD_CORR_MSK]
                .where(cloud[cst.POINTS_CLOUD_CORR_MSK].values != 0)
                .count()
            )
        else:
            nb_points = cloud.shape[0]

        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(nb_points))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        if isinstance(cloud, xr.Dataset):
            for x_item, y_item, z_item, mask_item in zip(  # noqa: B905
                np.nditer(cloud[cst.X].values),
                np.nditer(cloud[cst.Y].values),
                np.nditer(cloud[cst.Z].values),
                np.nditer(cloud[cst.POINTS_CLOUD_CORR_MSK].values),
            ):
                if mask_item != 0:
                    ply_file.write("{} {} {}\n".format(x_item, y_item, z_item))
        else:
            for xyz in cloud.itertuples():
                ply_file.write(
                    "{} {} {}\n".format(
                        getattr(xyz, cst.X),
                        getattr(xyz, cst.Y),
                        getattr(xyz, cst.Z),
                    )
                )


def write_vector(polys, path_to_file, epsg, driver="GPKG"):
    """
    Write list of polygons in a single vector file

    :param polys: list of polygons to write in the file
    :param path_to_file: file to create
    :param epsg: EPSG code of the polygons
    :param driver: vector file type (default format is geopackage)
    """
    crs = from_epsg(epsg)  # pylint: disable=c-extension-no-member
    sch = {"geometry": "Polygon", "properties": {"Type": "str:10"}}

    with fiona.open(
        path_to_file, "w", crs=crs, driver=driver, schema=sch
    ) as vector_file:
        for poly in polys:
            poly_dict = {
                "geometry": mapping(poly),
                "properties": {"Type": "Polygon"},
            }
            vector_file.write(poly_dict)


def rasterio_write_georaster(
    raster_file: str,
    data: np.ndarray,
    profile: dict = None,
    window: rio.windows.Window = None,
    descriptor=None,
    bands_description=None,
):
    """
    Write a raster file from array

    :param raster_file: Image file
    :param data: image data
    :param profile: rasterio profile
    """

    def write_data(data, window=None, descriptor=None):
        """
        Write data through descriptor
        :param data: data to write on disk
        :param window: window
        :param descriptor: descriptor
        :return:
        """

        if len(data.shape) == 2:
            descriptor.write(data, 1, window=window)
        elif len(data.shape) > 2:
            if data.shape[2] < 5:
                # wrong convention : cols, rows, bands
                # revert axis to bands, cols, rows
                data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])

            descriptor.write(data, window=window)

    if descriptor is not None:
        if bands_description is not None:
            for idx, description in enumerate(bands_description):
                # Band indexing starts at 1
                descriptor.set_band_description(idx + 1, str(description))
                descriptor.write_band(idx + 1, data[idx, :, :], window=window)
        else:
            write_data(data, window=window, descriptor=descriptor)

    else:
        count = 1
        width, height = data.shape[0], data.shape[1]
        if len(data.shape) > 2:
            count = data.shape[0]
            width, height = data.shape[1], data.shape[2]

        if profile is None:
            profile = DefaultGTiffProfile(count=count)
            profile["height"] = height
            profile["width"] = width
            profile["dtype"] = "float32"

        with rio.open(raster_file, "w", **profile) as new_descriptor:
            write_data(data, window=window, descriptor=new_descriptor)

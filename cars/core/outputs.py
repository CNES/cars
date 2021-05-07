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
import logging
import os
from typing import Union

# Third party imports
import fiona
import numpy as np
import pandas
import xarray as xr
import yaml
from fiona.crs import from_epsg
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

    with open(path_ply_file, "w") as ply_file:
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
            for x_item, y_item, z_item, mask_item in zip(
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
    crs = from_epsg(epsg)
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


def write_dask_config(dask_config: dict, output_dir: str, file_name: str):
    """
    Writes the dask config used in yaml format.

    :param dask_config: Dask config used
    :type dask_config: dict
    :param output_dir: output directory path
    :type dask_config: dict
    :param output_dir: output directory path
    """

    # warning
    logging.info(
        "Dask will merge several config files"
        "located at default locations such as"
        " ~/.config/dask/ .\n Dask config in "
        " $DASK_DIR will be used with the highest priority."
    )

    # file path where to store the dask config
    dask_config_path = os.path.join(output_dir, file_name + ".yaml")
    with open(dask_config_path, "w") as dask_config_file:
        yaml.dump(dask_config, dask_config_file)

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
Output module:
contains all compute_dsm.py main pipeline reading/writing functions
"""

#TODO ce fichier ne devrait il pas etre plus "proche" de compute_dsm ?

# Standard imports
from typing import Tuple
import logging
from contextlib import contextmanager

# Third party imports
import os
import numpy as np
from affine import Affine
from tqdm import tqdm
import rasterio as rio
import xarray as xr
from dask.distributed import as_completed

from cars import constants as cst


def compute_output_window(tile, full_bounds, resolution):
    """
    Computes destination indices slice of a given tile.

    :param tile: raster tile
    :type tile: xarray.Dataset
    :param full_bounds: bounds of the full output image ordered as the
        following: x_min, y_min, x_max, y_max.
    :type full_bounds: tuple(float, float, float, float).
    :param resolution: output resolution.
    :type resolution: float.
    :return: slices indices as i_xmin, i_ymin, i_xmax, i_ymax.
    :rtype: tuple(int, int, int, int)
    """
    x_min, _, _, y_max = full_bounds

    x_0 = np.round((np.min(tile.coords[cst.X]) - x_min) / resolution - 0.5)
    y_0 = np.round((y_max - np.max(tile.coords[cst.Y])) / resolution - 0.5)
    x_1 = np.round((np.max(tile.coords[cst.X]) - x_min) / resolution - 0.5)
    y_1 = np.round((y_max - np.min(tile.coords[cst.Y])) / resolution - 0.5)

    return (int(x_0), int(y_0), int(x_1), int(y_1))


@contextmanager
def rasterio_handles(names, files, params, nodata_values, nb_bands):
    """
    Open a context containing a series of rasterio handles. All input
    lists. Must have the same length.

    :param names: List of names to index the output dictionnary
    :type names: list
    :param files: List of path to files
    :type files: List
    :param params: List of rasterio parameters as dictionaries
    :type params: List
    :param nodata_values: List of nodata values
    :type nodata_values: List
    :param nb_bands: List of number of bands
    :type nb_bands: List
    :return: A dicionary of rasterio handles
        that can be used as a context manager, indexed by names
    :rtype: Dict
    """
    file_handles = {}
    for name_item, file_item, params_item, nodata_item, nb_bands_item in zip(
            names, files, params, nodata_values, nb_bands):

        file_handles[name_item] = rio.open(file_item, 'w',
                                            count=nb_bands_item,
                                            nodata=nodata_item,
                                            **params_item)
    try:
        yield file_handles
    finally:
        for handle in file_handles.values():
            handle.close()


def write_geotiff_dsm(future_dsm,
        output_dir: str,
        x_size: int,
        y_size: int,
        bounds: Tuple[float, float, float, float],
        resolution: float,
        epsg: int,
        nb_bands: int,
        dsm_no_data: float,
        color_no_data: float,
        write_color: bool = True,
        color_dtype: np.dtype=np.float32,
        write_stats: bool = False,
        write_msk = False,
        msk_no_data: int = 65535,
        prefix: str=""):
    """
    Writes result tiles to GTiff file(s).

    :param future_dsm: iterable containing future output tiles.
    :type future_dsm: list(dask.future)
    :param output_dir: output directory path
    :param x_size: full output x size.
    :param y_size: full output y size.
    :param bounds: geographic bounds of the tile (xmin, ymin, xmax, ymax).
    :param resolution: resolution of the tiles.
    :param epsg: epsg numeric code of the output tiles.
    :param nb_bands: number of band in the color layer.
    :param dsm_no_data: value to fill no data in height layer.
    :param color_no_data: value to fill no data in color layer(s).
    :param write_color: bolean enabling the ortho-image's writting
    :param color_dtype: type to use for the ortho-image
    :param write_stats: bolean enabling the rasterization statistics' writting
    :param write_msk: boolean enabling the rasterized mask's writting
    :param msk_no_data: no data to use in for the rasterized mask
    :param prefix: written filenames prefix

    """
    geotransform = (bounds[0], resolution, 0.0, bounds[3], 0.0, -resolution)
    transform = Affine.from_gdal(*geotransform)

    # common parameters for rasterio output
    dsm_rio_params = dict(
        height=y_size, width=x_size, driver='GTiff', dtype=np.float32,
        transform=transform, crs='EPSG:{}'.format(epsg), tiled=True
    )
    clr_rio_params = dict(
        height=y_size, width=x_size, driver='GTiff', dtype=color_dtype,
        transform=transform, crs='EPSG:{}'.format(epsg), tiled=True
    )

    dsm_rio_params_uint16 = dict(
        height=y_size, width=x_size, driver='GTiff', dtype=np.uint16,
        transform=transform, crs='EPSG:{}'.format(epsg), tiled=True
    )

    msk_rio_params_uint16 = dict(
        height=y_size, width=x_size, driver='GTiff', dtype=np.uint16,
        transform=transform, crs='EPSG:{}'.format(epsg), tiled=True
    )


    dsm_file = os.path.join(output_dir, prefix+'dsm.tif')

    # Prepare values for file handles
    names = ['dsm']
    files = [dsm_file]
    params = [dsm_rio_params]
    nodata_values = [dsm_no_data]
    nb_bands_to_write = [1]

    if write_color:
        names.append('clr')
        clr_file = os.path.join(output_dir, prefix + 'clr.tif')
        files.append(clr_file)
        params.append(clr_rio_params)
        nodata_values.append(color_no_data)
        nb_bands_to_write.append(nb_bands)

    if write_stats:
        names.append('dsm_mean')
        dsm_mean_file = os.path.join(output_dir, prefix + 'dsm_mean.tif')
        files.append(dsm_mean_file)
        params.append(dsm_rio_params)
        nodata_values.append(dsm_no_data)
        nb_bands_to_write.append(1)
        names.append('dsm_std')
        dsm_std_file = os.path.join(output_dir, prefix + 'dsm_std.tif')
        files.append(dsm_std_file)
        params.append(dsm_rio_params)
        nodata_values.append(dsm_no_data)
        nb_bands_to_write.append(1)

        names.append('dsm_n_pts')
        dsm_n_pts_file = os.path.join(output_dir, prefix + 'dsm_n_pts.tif')
        files.append(dsm_n_pts_file)
        params.append(dsm_rio_params_uint16)
        nodata_values.append(0)
        nb_bands_to_write.append(1)
        names.append('dsm_pts_in_cell')
        dsm_pts_in_cell_file = \
            os.path.join(output_dir, prefix + 'dsm_pts_in_cell.tif')
        files.append(dsm_pts_in_cell_file)
        params.append(dsm_rio_params_uint16)
        nodata_values.append(0)
        nb_bands_to_write.append(1)

    if write_msk:
        names.append('msk')
        msk_file = os.path.join(output_dir, prefix + 'msk.tif')
        files.append(msk_file)
        params.append(msk_rio_params_uint16)
        nodata_values.append(msk_no_data)
        nb_bands_to_write.append(1)

    # detect if we deal with dask.future or plain datasets
    has_datasets = True
    for tile in future_dsm:
        if tile is None:
            continue
        has_datasets = has_datasets and isinstance(tile, xr.Dataset)

    # get file handle(s) with optional color file.
    with rasterio_handles(names, files, params,
                            nodata_values, nb_bands_to_write) as rio_handles:

        def write(raster_tile):
            """
            Inner function for tiles writing
            """

            # Skip empty tile
            if raster_tile is None:
                logging.debug('Ignoring empty tile')
                return

            x_0, y_0, x_1, y_1 = compute_output_window(raster_tile, bounds,
                                                   resolution)

            logging.debug(
                "Writing tile of size [{}, {}] at index [{}, {}]"
                    .format(x_1 - x_0 + 1, y_1 - y_0 + 1, x_0, y_0)
            )

            # window is speficied as origin & size
            window = rio.windows.Window(x_0, y_0, x_1 - x_0 + 1, y_1 - y_0 + 1)

            rio_handles['dsm'].write_band(1,
                            raster_tile[cst.RASTER_HGT].values, window=window)

            if write_color:
                rio_handles['clr'].write(raster_tile[
                            cst.RASTER_COLOR_IMG].values.astype(color_dtype),
                                window=window)

            if write_stats:
                rio_handles['dsm_mean'].write_band(1,
                                raster_tile[cst.RASTER_HGT_MEAN].values,
                                window=window)
                rio_handles['dsm_std'].write_band(1,
                                raster_tile[cst.RASTER_HGT_STD_DEV].values,
                                window=window)
                rio_handles['dsm_n_pts'].write_band(1,
                                raster_tile[cst.RASTER_NB_PTS].values,
                                window=window)
                rio_handles['dsm_pts_in_cell'].write_band(1,
                                raster_tile[cst.RASTER_NB_PTS_IN_CELL].values,
                                window=window)

            ds_values_list = [key for key, _ in raster_tile.items()]
            if cst.RASTER_MSK in ds_values_list and write_msk:
                rio_handles['msk'].write_band(1,
                                raster_tile[cst.RASTER_MSK].values,
                                window=window)

        # Multiprocessing mode
        if has_datasets:
            for raster_tile in future_dsm:
                write(raster_tile)

        # dask mode
        else:
            for future, raster_tile in tqdm(
                    as_completed(future_dsm, with_results=True),
                    total=len(future_dsm), desc="Writing output tif file"):

                write(raster_tile)

                logging.debug('Waiting for next tile')
                if future is not None:
                    future.cancel()

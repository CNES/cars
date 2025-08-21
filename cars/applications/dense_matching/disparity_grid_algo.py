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
this module contains the wrapper used in disparity grid computation.
"""

# Standard imports
import itertools
import logging

# Third party imports
import affine
import numpy as np
import rasterio
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter
from shareloc.proj_utils import transform_physical_point_to_index

# CARS imports
import cars.applications.dense_matching.dense_matching_constants as dm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications.dense_matching.dense_matching_algo import (
    LinearInterpNearestExtrap,
)
from cars.core import inputs, projection
from cars.core.projection import point_cloud_conversion
from cars.data_structures import cars_dataset, cars_dict


def generate_disp_grids_dataset(
    grid_min,
    grid_max,
    saving_info,
    raster_profile,
    window=None,
    row_coords=None,
    col_coords=None,
):
    """
    Generate disparity grids xarray dataset

    :param grid_min: disp grid min
    :type grid_min: np.ndarray
    :param grid_max: disp grid max
    :type grid_max: np.ndarray
    :param saving_info: saving infos
    :type saving_info: dict
    :param raster_profile: raster_profile
    :type raster_profile: dict
    :param row_coords: row cooordinates
    :type row_coords: np.ndarray, optional
    :param col_coords: col coordinates
    :type col_coords: np.ndarray, optional

    :return: disp range dataset
    :rtype: xarray.Dataset
    """

    if row_coords is None:
        row_coords = np.arange(0, grid_min.shape[0])

    if col_coords is None:
        col_coords = np.arange(0, grid_min.shape[1])

    disp_range_tile = xr.Dataset(
        data_vars={
            dm_cst.DISP_MIN_GRID: (["row", "col"], grid_min),
            dm_cst.DISP_MAX_GRID: (["row", "col"], grid_max),
        },
        coords={
            "row": row_coords,
            "col": col_coords,
        },
    )

    cars_dataset.fill_dataset(
        disp_range_tile,
        saving_info=saving_info,
        window=window,
        profile=raster_profile,
        attributes=None,
        overlaps=None,
    )

    return disp_range_tile


def generate_disp_range_const_tile_wrapper(
    row_range,
    col_range,
    dmin,
    dmax,
    raster_profile,
    saving_info,
    saving_info_global_infos,
):
    """
    Generate disparity range dataset from constant dmin and dmax

    :param row_range: Row range
    :type row_range: list
    :param col_range: Column range.
    :type col_range: list
    :param dmin:  disparity minimum.
    :type dmin: float
    :param dmax: disparity maximum.
    :type dmax: float
    :param raster_profile: The raster profile.
    :type raster_profile: dict
    :param saving_info: The disp range grid saving information.
    :type saving_info: dict
    :param saving_info_global_infos: Global info saving infos.
    :type saving_info_global_infos: dict

    :return: Disparity range grid
    :rtype: dict
    """
    grid_min = np.empty((len(row_range), len(col_range)))
    grid_max = np.empty((len(row_range), len(col_range)))
    grid_min[:, :] = dmin
    grid_max[:, :] = dmax

    mono_tile_saving_info = ocht.update_saving_infos(saving_info, row=0, col=0)
    disp_range = generate_disp_grids_dataset(
        grid_min, grid_max, mono_tile_saving_info, raster_profile, window=None
    )

    # Generate infos on global min and max
    global_infos = cars_dict.CarsDict(
        {"global_min": np.nanmin(dmin), "global_max": np.nanmin(dmax)}
    )
    cars_dataset.fill_dict(global_infos, saving_info=saving_info_global_infos)

    return disp_range, global_infos


def generate_disp_range_from_dem_wrapper(
    epipolar_grid_array_window,
    full_epi_row_range,
    full_epi_col_range,
    sensor_image_right,
    grid_right,
    geom_plugin_with_dem_and_geoid,
    dem_median,
    dem_min,
    dem_max,
    altitude_delta_min,
    altitude_delta_max,
    raster_profile,
    saving_info,
    saving_info_global_infos,
    filter_overlap,
    disp_to_alt_ratio,
    disp_min_threshold=None,
    disp_max_threshold=None,
):
    """
    Generate disparity range dataset from dems

    :param epipolar_grid_array_window: The window of the epipolar grid array.
    :type epipolar_grid_array_window: dict
    :param full_epi_row_range: The full range of rows in the epipolar grid.
    :type full_epi_row_range: list
    :param full_epi_col_range: The full range of columns in the epipolar grid.
    :type full_epi_col_range: list
    :param sensor_image_right: The right sensor image.
    :type sensor_image_right:  dict
    :param grid_right: The right epipolar grid.
    :type grid_right: dict
    :param geom_plugin_with_dem_and_geoid: The geometry plugin with DEM.
    :type geom_plugin_with_dem_and_geoid: object
    :param dem_median: Path of dem median.
    :type dem_median: str
    :param dem_min: Path of dem min.
    :type dem_min: str
    :param dem_max: Path of dem max.
    :type dem_max: srt
    :param altitude_delta_min: The minimum altitude delta.
    :type altitude_delta_min: float
    :param altitude_delta_max: The maximum altitude delta.
    :type altitude_delta_max: float
    :param raster_profile: The raster profile.
    :type raster_profile: dict
    :param saving_info: The disp range grid saving information.
    :type saving_info: dict
    :param saving_info_global_infos: Global info saving infos.
    :type saving_info_global_infos: dict
    :param filter_overlap: The overlap to use for filtering.
    :type filter_overlap: int
    :param disp_to_alt_ratio: disparity to altitude ratio
    :type disp_to_alt_ratio: float
    :param disp_min_threshold: The minimum disparity threshold.
    :type disp_min_threshold: float, optional
    :param disp_max_threshold: The maximum disparity threshold.
    :type disp_max_threshold: float, optional

    :return: Disparity range grid
    :rtype: dict
    """

    # Geometry plugin
    geo_plugin = geom_plugin_with_dem_and_geoid

    # get epsg
    terrain_epsg = inputs.rasterio_get_epsg(dem_median)

    # Get epipolar position of all dem mean
    transform_dem_median = inputs.rasterio_get_transform(dem_median)

    # use local disparity

    # Get associated alti mean / min / max values
    dem_median_shape = inputs.rasterio_get_size(dem_median)
    dem_median_width, dem_median_height = dem_median_shape

    # get corresponding window from epipolar_array_window
    epi_grid_margin = filter_overlap + 1
    epi_grid_row_min = epipolar_grid_array_window["row_min"]
    epi_grid_row_max = epipolar_grid_array_window["row_max"]
    epi_grid_col_min = epipolar_grid_array_window["col_min"]
    epi_grid_col_max = epipolar_grid_array_window["col_max"]

    def clip(value, min_value, max_value):
        """
        Clip a value inside bounds
        """
        return int(max(min_value, min(value, max_value)))

    # Epi grid tile coordinate to use, with and without margins
    epi_grid_row_min_with_margin = clip(
        epi_grid_row_min - epi_grid_margin, 0, len(full_epi_row_range)
    )
    epi_grid_row_max_with_margin = clip(
        epi_grid_row_max + epi_grid_margin, 0, len(full_epi_row_range)
    )
    epi_grid_col_min_with_margin = clip(
        epi_grid_col_min - epi_grid_margin, 0, len(full_epi_col_range)
    )
    epi_grid_col_max_with_margin = clip(
        epi_grid_col_max + epi_grid_margin, 0, len(full_epi_col_range)
    )

    # range to use for epipolar interpolation
    row_range_with_margin = full_epi_row_range[
        epi_grid_row_min_with_margin:epi_grid_row_max_with_margin
    ]
    row_range_no_margin = full_epi_row_range[epi_grid_row_min:epi_grid_row_max]
    col_range_with_margin = full_epi_col_range[
        epi_grid_col_min_with_margin:epi_grid_col_max_with_margin
    ]
    col_range_no_margin = full_epi_col_range[epi_grid_col_min:epi_grid_col_max]

    # Loc on dem median
    epi_bbox = [
        (np.min(col_range_with_margin), np.min(row_range_with_margin)),
        (np.min(col_range_with_margin), np.max(row_range_with_margin)),
        (np.max(col_range_with_margin), np.min(row_range_with_margin)),
        (np.max(col_range_with_margin), np.max(row_range_with_margin)),
    ]
    sensor_bbox = geo_plugin.sensor_position_from_grid(grid_right, epi_bbox)
    transform_sensor = inputs.rasterio_get_transform(
        sensor_image_right["image"]["main_file"]
    )
    row_sensor_bbox, col_sensor_bbox = transform_physical_point_to_index(
        ~transform_sensor, sensor_bbox[:, 1], sensor_bbox[:, 0]
    )

    terrain_bbox = geo_plugin.direct_loc(
        sensor_image_right["image"]["main_file"],
        sensor_image_right["geomodel"],
        col_sensor_bbox,
        row_sensor_bbox,
    )

    # reshape terrain bbox
    terrain_bbox = terrain_bbox[0:2].T
    terrain_bbox[:, [1, 0]] = terrain_bbox[:, [0, 1]]

    # get pixel location on dem median
    pixel_roi_dem_mean = inputs.rasterio_get_pixel_points(
        dem_median, terrain_bbox
    )

    # Add margins (for interpolation) and clip
    dem_margin = 10  # arbitrary
    roi_lower_row = np.floor(np.min(pixel_roi_dem_mean[:, 0])) - dem_margin
    roi_upper_row = np.ceil(np.max(pixel_roi_dem_mean[:, 0])) + dem_margin
    roi_lower_col = np.floor(np.min(pixel_roi_dem_mean[:, 1])) - dem_margin
    roi_upper_col = np.ceil(np.max(pixel_roi_dem_mean[:, 1])) + dem_margin

    min_row = clip(roi_lower_row, 0, dem_median_height)
    max_row = clip(roi_upper_row, 0, dem_median_height)
    min_col = clip(roi_lower_col, 0, dem_median_width)
    max_col = clip(roi_upper_col, 0, dem_median_width)

    # compute terrain positions to use (all dem min and max)
    row_indexes = range(min_row, max_row)
    col_indexes = range(min_col, max_col)
    transformer = rasterio.transform.AffineTransformer(transform_dem_median)

    indexes = np.array(list(itertools.product(row_indexes, col_indexes)))

    row = indexes[:, 0]
    col = indexes[:, 1]
    x_mean, y_mean = transformer.xy(row, col)
    terrain_positions = np.transpose(np.array([x_mean, y_mean]))

    # dem mean in terrain_epsg
    x_mean = terrain_positions[:, 0]
    y_mean = terrain_positions[:, 1]

    dem_median_list = inputs.rasterio_get_values(
        dem_median, x_mean, y_mean, point_cloud_conversion
    )

    nan_mask = ~np.isnan(dem_median_list)

    # transform to lon lat
    terrain_position_lon_lat = projection.point_cloud_conversion(
        terrain_positions, terrain_epsg, 4326
    )
    lon_mean = terrain_position_lon_lat[:, 0]
    lat_mean = terrain_position_lon_lat[:, 1]

    if None not in (dem_min, dem_max, dem_median):
        # dem min and max are in 4326
        dem_min_list = inputs.rasterio_get_values(
            dem_min, lon_mean, lat_mean, point_cloud_conversion
        )
        dem_max_list = inputs.rasterio_get_values(
            dem_max, lon_mean, lat_mean, point_cloud_conversion
        )
        nan_mask = nan_mask & ~np.isnan(dem_min_list) & ~np.isnan(dem_max_list)
    else:
        dem_min_list = dem_median_list - altitude_delta_min
        dem_max_list = dem_median_list + altitude_delta_max

    # filter nan value from input points
    lon_mean = lon_mean[nan_mask]
    lat_mean = lat_mean[nan_mask]
    dem_median_list = dem_median_list[nan_mask]
    dem_min_list = dem_min_list[nan_mask]
    dem_max_list = dem_max_list[nan_mask]

    # sensors physical positions
    (
        ind_cols_sensor,
        ind_rows_sensor,
        _,
    ) = geom_plugin_with_dem_and_geoid.inverse_loc(
        sensor_image_right["image"]["main_file"],
        sensor_image_right["geomodel"],
        lat_mean,
        lon_mean,
        z_coord=dem_median_list,
    )

    # Generate epipolar disp grids
    # Get epipolar positions
    (epipolar_positions_row, epipolar_positions_col) = np.meshgrid(
        col_range_with_margin,
        row_range_with_margin,
    )
    epipolar_positions = np.stack(
        [epipolar_positions_row, epipolar_positions_col], axis=2
    )

    # Get sensor position
    sensors_positions = (
        geom_plugin_with_dem_and_geoid.sensor_position_from_grid(
            grid_right,
            np.reshape(
                epipolar_positions,
                (
                    epipolar_positions.shape[0] * epipolar_positions.shape[1],
                    2,
                ),
            ),
        )
    )

    # compute reverse matrix
    transform_sensor = rasterio.Affine(
        *np.abs(
            inputs.rasterio_get_transform(
                sensor_image_right["image"]["main_file"]
            )
        )
    )

    trans_inv = ~transform_sensor
    # Transform to positive values
    trans_inv = np.array(trans_inv)
    trans_inv = np.reshape(trans_inv, (3, 3))
    if trans_inv[0, 0] < 0:
        trans_inv[0, :] *= -1
    if trans_inv[1, 1] < 0:
        trans_inv[1, :] *= -1
    trans_inv = affine.Affine(*list(trans_inv.flatten()))

    # Transform physical position to index
    index_positions = np.empty(sensors_positions.shape)
    for row_point in range(index_positions.shape[0]):
        row_geo, col_geo = sensors_positions[row_point, :]
        col, row = trans_inv * (row_geo, col_geo)
        index_positions[row_point, :] = (row, col)

    ind_rows_sensor_grid = index_positions[:, 0] - 0.5
    ind_cols_sensor_grid = index_positions[:, 1] - 0.5

    if len(ind_rows_sensor) < 5:
        # QH6214 needs at least 4 points for interpolation

        grid_min = np.empty(
            (len(row_range_no_margin), len(col_range_no_margin))
        )
        grid_max = np.empty(
            (len(row_range_no_margin), len(col_range_no_margin))
        )
        grid_min[:, :] = 0
        grid_max[:, :] = 0

        disp_range = generate_disp_grids_dataset(
            grid_min,
            grid_max,
            saving_info,
            raster_profile,
            window=epipolar_grid_array_window,
            row_coords=row_range_no_margin,
            col_coords=col_range_no_margin,
        )

        # Generate infos on global min and max
        global_infos = cars_dict.CarsDict(
            {
                "global_min": 0,
                "global_max": 0,
            }
        )
        cars_dataset.fill_dict(
            global_infos, saving_info=saving_info_global_infos
        )

        return disp_range, global_infos

    # Interpolate disparity
    disp_min_points = -(dem_max_list - dem_median_list) / disp_to_alt_ratio
    disp_max_points = -(dem_min_list - dem_median_list) / disp_to_alt_ratio

    interp_min_linear = LinearInterpNearestExtrap(
        list(zip(ind_rows_sensor, ind_cols_sensor)),  # noqa: B905
        disp_min_points,
    )
    interp_max_linear = LinearInterpNearestExtrap(
        list(zip(ind_rows_sensor, ind_cols_sensor)),  # noqa: B905
        disp_max_points,
    )

    grid_min = np.reshape(
        interp_min_linear(ind_rows_sensor_grid, ind_cols_sensor_grid),
        (
            epipolar_positions.shape[0],
            epipolar_positions.shape[1],
        ),
    )

    grid_max = np.reshape(
        interp_max_linear(ind_rows_sensor_grid, ind_cols_sensor_grid),
        (
            epipolar_positions.shape[0],
            epipolar_positions.shape[1],
        ),
    )

    # Add margin
    diff = grid_max - grid_min
    logging.info("Max grid max - grid min : {} disp ".format(np.max(diff)))

    if disp_min_threshold is not None:
        if np.any(grid_min < disp_min_threshold):
            logging.warning(
                "Override disp_min  with disp_min_threshold {}".format(
                    disp_min_threshold
                )
            )
            grid_min[grid_min < disp_min_threshold] = disp_min_threshold
    if disp_max_threshold is not None:
        if np.any(grid_max > disp_max_threshold):
            logging.warning(
                "Override disp_max with disp_max_threshold {}".format(
                    disp_max_threshold
                )
            )
            grid_max[grid_max > disp_max_threshold] = disp_max_threshold

    # generate footprint
    footprint_mask = create_circular_mask(filter_overlap, filter_overlap)
    grid_min = minimum_filter(
        grid_min, footprint=footprint_mask, mode="nearest"
    )
    grid_max = maximum_filter(
        grid_max, footprint=footprint_mask, mode="nearest"
    )

    # Create xarray dataset
    disp_range = generate_disp_grids_dataset(
        grid_min,
        grid_max,
        saving_info,
        raster_profile,
        window=epipolar_grid_array_window,
        row_coords=row_range_with_margin,
        col_coords=col_range_with_margin,
    )

    # crop epipolar grid from margin added for propagation filter
    disp_range = disp_range.sel(
        row=list(row_range_no_margin), col=list(col_range_no_margin)
    )

    # Generate infos on global min and max
    global_infos = cars_dict.CarsDict(
        {
            "global_min": np.floor(np.nanmin(grid_min)),
            "global_max": np.ceil(np.nanmax(grid_max)),
        }
    )
    cars_dataset.fill_dict(global_infos, saving_info=saving_info_global_infos)

    return disp_range, global_infos


def create_circular_mask(height, width):
    """
    Create a circular mask for footprint around pixel

    :param height: height of footprint
    :type height: int
    :param width: width of footprint
    :type width: int

    :return: mask representing circular footprint
    :rtype: np.ndarray
    """
    center = (int(width / 2), int(height / 2))
    radius = min(center[0], center[1], width - center[0], height - center[1])

    y_grid, x_grid = np.ogrid[:height, :width]
    dist_from_center = np.sqrt(
        (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2
    )

    mask = dist_from_center <= radius
    return mask.astype(bool)

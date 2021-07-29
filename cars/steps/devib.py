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
Devib module:
contains functions for DEM devibration step in prepare and compute_dsm pipeline.
"""

import logging
import math

# Standard imports
from typing import Tuple

# Third party imports
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

# CARS imports
from cars.conf import input_parameters
from cars.core import projection


def lowres_initial_dem_splines_fit(
    lowres_dsm_from_matches: xr.Dataset,
    lowres_initial_dem: xr.Dataset,
    origin: np.ndarray,
    time_direction_vector: np.ndarray,
    ext: int = 3,
    order: int = 3,
):
    """
    This function takes 2 datasets containing DSM and models the
    difference between the two as an UnivariateSpline along the
    direction given by origin and time_direction_vector. Internally,
    it looks for the highest smoothing factor that satisfies the rmse threshold.

    :param lowres_dsm_from_matches: Dataset containing the low resolution DSM
        obtained from matches, as returned by the
        rasterization.simple_rasterization_dataset function.
    :param lowres_initial_dem: Dataset containing the low resolution DEM,
        obtained by otb_pipelines.read_lowres_dem function,
        on the same grid as lowres_dsm_from_matches
    :param origin: coordinates of origin point for line
    :type origin: list(float) or np.array(float) of size 2
    :param time_direction_vector: direction vector of line
    :type time_direction_vector: list(float) or np.array(float) of size 2
    :param ext: behavior outside of interpolation domain
    :param order: spline order
    """
    # Initial DSM difference
    dsm_diff = lowres_initial_dem.hgt - lowres_dsm_from_matches.hgt

    # Form arrays of coordinates
    x_values_2d, y_values_2d = np.meshgrid(dsm_diff.x, dsm_diff.y)

    # Project coordinates on the line
    # of increasing acquisition time to form 1D coordinates
    # If there are sensor oscillations, they will occur in this direction
    linear_coords = projection.project_coordinates_on_line(
        x_values_2d, y_values_2d, origin, time_direction_vector
    )

    # Form a 1D array with diff values indexed with linear coords
    linear_diff_array = xr.DataArray(
        dsm_diff.values.ravel(), coords={"l": linear_coords.ravel()}, dims=("l")
    )
    linear_diff_array = linear_diff_array.dropna(dim="l")
    linear_diff_array = linear_diff_array.sortby("l")

    # Apply median filtering within cell matching input dsm resolution
    min_l = np.min(linear_diff_array.l)
    max_l = np.max(linear_diff_array.l)
    nbins = int(
        math.ceil((max_l - min_l) / (lowres_dsm_from_matches.resolution))
    )
    median_linear_diff_array = linear_diff_array.groupby_bins(
        "l", nbins
    ).median()
    median_linear_diff_array = median_linear_diff_array.rename({"l_bins": "l"})
    median_linear_diff_array = median_linear_diff_array.assign_coords(
        {"l": np.array([d.mid for d in median_linear_diff_array.l.data])}
    )

    count_linear_diff_array = linear_diff_array.groupby_bins("l", nbins).count()
    count_linear_diff_array = count_linear_diff_array.rename({"l_bins": "l"})
    count_linear_diff_array = count_linear_diff_array.assign_coords(
        {"l": np.array([d.mid for d in count_linear_diff_array.l.data])}
    )

    # Filter measurements with insufficient amount of points
    median_linear_diff_array = median_linear_diff_array.where(
        count_linear_diff_array > 100
    ).dropna(dim="l")

    if len(median_linear_diff_array) < 100:
        logging.warning(
            "Insufficient amount of points along time direction "
            "after measurements filtering to estimate correction "
            "to fit initial DEM"
        )
        return None

    # Apply butterworth lowpass filter to retrieve only the low frequency
    # (from example of doc: https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.signal.butter.html#scipy.signal.butter )
    b, a = butter(3, 0.05)
    zi_filter = lfilter_zi(b, a)
    z_filter, _ = lfilter(
        b,
        a,
        median_linear_diff_array.values,
        zi=zi_filter * median_linear_diff_array.values[0],
    )
    lfilter(b, a, z_filter, zi=zi_filter * z_filter[0])
    filtered_median_linear_diff_array = xr.DataArray(
        filtfilt(b, a, median_linear_diff_array.values),
        coords=median_linear_diff_array.coords,
    )

    # Estimate performances of spline s = 100 * length of diff array
    smoothing_factor = 100 * len(filtered_median_linear_diff_array.l)
    splines = interpolate.UnivariateSpline(
        filtered_median_linear_diff_array.l,
        filtered_median_linear_diff_array.values,
        ext=ext,
        k=order,
        s=smoothing_factor,
    )
    estimated_correction = xr.DataArray(
        splines(filtered_median_linear_diff_array.l),
        coords=filtered_median_linear_diff_array.coords,
    )
    rmse = (filtered_median_linear_diff_array - estimated_correction).std(
        dim="l"
    )

    target_rmse = 0.3

    # If RMSE is too high, try to decrease smoothness until it fits
    while rmse > target_rmse and smoothing_factor > 0.001:
        # divide s by 2
        smoothing_factor /= 2

        # Estimate splines
        splines = interpolate.UnivariateSpline(
            filtered_median_linear_diff_array.l,
            filtered_median_linear_diff_array.values,
            ext=ext,
            k=order,
            s=smoothing_factor,
        )

        # Compute RMSE
        estimated_correction = xr.DataArray(
            splines(filtered_median_linear_diff_array.l),
            coords=filtered_median_linear_diff_array.coords,
        )

        rmse = (filtered_median_linear_diff_array - estimated_correction).std(
            dim="l"
        )

    logging.debug(
        "Best smoothing factor for splines regression: "
        "{} (rmse={})".format(smoothing_factor, rmse)
    )

    # Return estimated spline
    return splines


def acquisition_direction(conf, dem: str) -> Tuple[np.ndarray]:
    """
    Computes the mean acquisition of the input images pair

    :param conf: cars input configuration dictionary
    :param dem: path to the dem directory
    :return: a tuple composed of :
        - the mean acquisition direction as a numpy array
        - the acquisition direction of the first product in the configuration
        as a numpy array
        - the acquisition direction of the second product in the configuration
        as a numpy array
    """
    vec1 = projection.get_time_ground_direction(
        conf, input_parameters.PRODUCT1_KEY, dem=dem
    )
    vec2 = projection.get_time_ground_direction(
        conf, input_parameters.PRODUCT2_KEY, dem=dem
    )
    time_direction_vector = (vec1 + vec2) / 2

    def display_angle(vec):
        """
        Display angle in degree from a vector x
        :param vec: vector to display
        :return: angle in degree
        """
        return 180 * math.atan2(vec[1], vec[0]) / math.pi

    logging.info(
        "Time direction average azimuth: "
        "{}° (img1: {}°, img2: {}°)".format(
            display_angle(time_direction_vector),
            display_angle(vec1),
            display_angle(vec2),
        )
    )

    return time_direction_vector, vec1, vec2

# !/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
this module contains the AuxiliaryFillingFromSensors application class.
"""


import numpy as np
import rasterio as rio
from scipy import interpolate
from shapely.geometry import Polygon

from cars.core.projection import polygon_projection


def fill_auxiliary(
    sensor_inputs,
    pairing,
    longitudes,
    latitudes,
    altitudes,
    geom_plugin,
    number_of_color_bands,
    number_of_classification_bands,
    color_interpolator,
    use_mask=False,
):
    """
    Compute color and classification for a list of points (lon, lat, alt) using
    information from sensor images

    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param pairing: pairing between input images
    :type pairing: list
    :param longitudes: list containing longitudes coordinates
    :type longitudes: list
    :param latitudes: list containing latitudes coordinates
    :type latitudes: list
    :param altitudes: list containing altitudes coordinates
    :type altitudes: list
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param number_of_color_bands: number of bands in the color image
    :type number_of_color_bands: int
    :param number_of_classification_bands: number of bands in the color image
    :type number_of_classification_bands: int
    :param color_interpolator: scipy interpolator use to interpolate color
        values
    :type color_interpolator: str
    :param use_mask: use mask information from sensors in color computation
    :type use_mask: bool

    """

    filled_color = np.zeros((number_of_color_bands, len(altitudes)))

    filled_classif = None
    if number_of_classification_bands:
        filled_classif = np.zeros(
            (number_of_classification_bands, len(altitudes)), dtype=bool
        )

    weights = np.zeros(len(altitudes))
    full_weights = np.zeros(len(altitudes))

    all_values = np.zeros((number_of_color_bands, len(altitudes)))

    for pair in pairing:

        first_sensor = sensor_inputs.get(pair[0])
        second_sensor = sensor_inputs.get(pair[1])

        # if first sensor has been filtered, use the second sensor instead
        if first_sensor is None:
            first_sensor = second_sensor
            second_sensor = None

        # process first sensor
        if first_sensor is not None:
            not_interpolated_mask, all_values_sensor = fill_from_one_sensor(
                first_sensor,
                filled_color,
                filled_classif,
                weights,
                longitudes,
                latitudes,
                altitudes,
                geom_plugin,
                number_of_color_bands,
                number_of_classification_bands,
                color_interpolator,
                use_mask=use_mask,
                return_all_points=True,
            )
            if all_values_sensor is not None:
                all_values += all_values_sensor
                full_weights += 1

            # process second sensor
            if second_sensor is not None:
                fill_from_one_sensor(
                    second_sensor,
                    filled_color,
                    filled_classif,
                    weights,
                    longitudes,
                    latitudes,
                    altitudes,
                    geom_plugin,
                    number_of_color_bands,
                    number_of_classification_bands,
                    color_interpolator,
                    not_interpolated_mask,
                    use_mask=use_mask,
                    return_all_points=False,
                )

    interpolated_pixels = filled_color != 0

    filled_color[:, interpolated_pixels[0, :]] /= weights[
        interpolated_pixels[0, :]
    ]
    filled_color[:, ~interpolated_pixels[0, :]] = (
        all_values[:, ~interpolated_pixels[0, :]]
        / full_weights[~interpolated_pixels[0, :]]
    )

    return filled_color, filled_classif


def fill_from_one_sensor(
    sensor,
    filled_color,
    filled_classif,
    weights,
    longitudes,
    latitudes,
    altitudes,
    geom_plugin,
    number_of_color_bands,
    number_of_classification_bands,
    color_interpolator,
    not_interpolated_mask=None,
    use_mask=False,
    return_all_points=False,
):
    """
    Compute color and classification contribution  for a list of points
    (lon, lat, alt) using information from a sensor image

    :param sensor: dictionary containing paths to input images and model
    :type sensor: dict
    :param filled_color: array containing (non normalized) color information
    :type filled_color: numpy.ndarray
    :param filled_classif: array containing classification information
    :type filled_classif: numpy.array
    :param weights: array containing weight for normalization
    :type weights: numpy.array
    :param longitudes: list containing longitudes coordinates
    :type longitudes: list
    :param latitudes: list containing latitudes coordinates
    :type latitudes: list
    :param altitudes: list containing altitudes coordinates
    :type altitudes: list
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param number_of_color_bands: number of bands in the color image
    :type number_of_color_bands: int
    :param number_of_classification_bands: number of bands in the color image
    :type number_of_classification_bands: int
    :param color_interpolator: scipy interpolator use to interpolate color
        values
    :type color_interpolator: str
    :param not_interpolated_mask: use mask information in color computation
    :type not_interpolated_mask: numpy.array
    :param use_mask: use mask information in color computation
    :type use_mask: bool
    :param return_all_points: compute interpolated values for all points
    :type return_all_points: bool

    """
    use_mask = True

    # Check if the sensor has color or classification
    reference_sensor_image = sensor.get(
        "color", sensor.get("classification", None)
    )

    output_not_interpolated_mask = np.ones(len(altitudes), dtype=bool)
    all_values = np.zeros((number_of_color_bands, len(altitudes)))

    # No filling information for this sensor, return
    if reference_sensor_image is None:
        return output_not_interpolated_mask, all_values
    # read metadata
    with rio.open(reference_sensor_image) as reference_image:
        sensor_height = reference_image.height
        sensor_width = reference_image.width

    # sensors physical positions
    (
        ind_cols_sensor,
        ind_rows_sensor,
        _,
    ) = geom_plugin.inverse_loc(
        reference_sensor_image,
        sensor["geomodel"],
        latitudes,
        longitudes,
        altitudes,
    )

    # Compute col and row bounds
    min_rows = np.min(ind_rows_sensor)
    max_rows = np.max(ind_rows_sensor)

    min_cols = np.min(ind_cols_sensor)
    max_cols = np.max(ind_cols_sensor)

    # Check for out of bound coordinates
    if (
        min_rows > sensor_height
        or max_rows < 0
        or min_cols > sensor_width
        or max_cols < 0
    ):
        return output_not_interpolated_mask, all_values

    if color_interpolator in ("linear", "nearest"):
        color_interpolator_margin = 1
    # TODO check parameter outside of thread
    else:
        raise RuntimeError(f"Invalid interpolator {color_interpolator}")

    # Classification interpolator is always nearest
    classif_interpolator = "nearest"
    classif_interpolator_margin = 1

    validity_mask = not_interpolated_mask
    if use_mask and sensor.get("mask"):
        with rio.open(sensor["mask"]) as sensor_mask_image:
            first_row = np.floor(max(min_rows - classif_interpolator_margin, 0))
            last_row = np.ceil(
                min(
                    max_rows + classif_interpolator_margin,
                    sensor_mask_image.height,
                )
            )
            first_col = np.floor(max(min_cols - classif_interpolator_margin, 0))
            last_col = np.ceil(
                min(
                    max_cols + classif_interpolator_margin,
                    sensor_mask_image.width,
                )
            )

            rio_window = rio.windows.Window.from_slices(
                (first_row, last_row),
                (first_col, last_col),
            )

            sensor_points = (
                np.arange(first_row, last_row),
                np.arange(first_col, last_col),
            )

            sensor_data = sensor_mask_image.read(1, window=rio_window)

            validity_mask = np.logical_not(
                interpolate.interpn(
                    sensor_points,
                    sensor_data,
                    (ind_rows_sensor, ind_cols_sensor),
                    bounds_error=False,
                    fill_value=1,
                    method=classif_interpolator,
                )
            )

            if not_interpolated_mask is not None:
                validity_mask = np.logical_and(
                    validity_mask, not_interpolated_mask
                )

    if sensor.get("color"):
        with rio.open(sensor["color"]) as sensor_color_image:
            # Only fill color if the number of bands matches. For exemple do
            # not try to fill an output RGB image from a greyscale "color"
            # image.
            if sensor_color_image.count == number_of_color_bands:
                first_row = np.floor(
                    max(np.min(ind_rows_sensor) - color_interpolator_margin, 0)
                )
                last_row = np.ceil(
                    min(
                        np.max(ind_rows_sensor) + color_interpolator_margin,
                        sensor_color_image.height,
                    )
                )
                first_col = np.floor(
                    max(np.min(ind_cols_sensor) - color_interpolator_margin, 0)
                )
                last_col = np.ceil(
                    min(
                        np.max(ind_cols_sensor) + color_interpolator_margin,
                        sensor_color_image.width,
                    )
                )

                rio_window = rio.windows.Window.from_slices(
                    (first_row, last_row),
                    (first_col, last_col),
                )

                sensor_points = (
                    np.arange(first_row, last_row),
                    np.arange(first_col, last_col),
                )

                nan_mask = np.zeros(len(altitudes), dtype=bool)
                for band in range(sensor_color_image.count):
                    # rio band convention
                    sensor_data = sensor_color_image.read(
                        band + 1, window=rio_window
                    )

                    if validity_mask is not None:
                        if return_all_points is True:
                            all_values[band, :] = interpolate.interpn(
                                sensor_points,
                                sensor_data,
                                (ind_rows_sensor, ind_cols_sensor),
                                bounds_error=False,
                                method=color_interpolator,
                            )
                            band_values = all_values[band, validity_mask]
                        # No need to interpolate on every points
                        else:
                            band_values = interpolate.interpn(
                                sensor_points,
                                sensor_data,
                                (
                                    ind_rows_sensor[validity_mask],
                                    ind_cols_sensor[validity_mask],
                                ),
                                bounds_error=False,
                                method=color_interpolator,
                            )
                        nan_values = np.isnan(band_values)
                        filled_color[band, validity_mask] += band_values
                        nan_mask[validity_mask] = np.logical_or(
                            nan_mask[validity_mask], nan_values
                        )
                    else:
                        band_values = interpolate.interpn(
                            sensor_points,
                            sensor_data,
                            (ind_rows_sensor, ind_cols_sensor),
                            bounds_error=False,
                            method=color_interpolator,
                        )
                        filled_color[band, :] += band_values
                        nan_mask = np.logical_or(
                            nan_mask, np.isnan(band_values)
                        )

                if validity_mask is not None:
                    output_not_interpolated_mask = np.logical_and(
                        ~validity_mask, nan_mask
                    )
                else:
                    output_not_interpolated_mask = nan_mask

                weights[validity_mask] += 1

    if filled_classif is not None and sensor.get("classification"):
        with rio.open(sensor["classification"]) as sensor_classif_image:

            if sensor_classif_image.count == number_of_classification_bands:
                first_row = np.floor(
                    max(
                        np.min(ind_rows_sensor) - classif_interpolator_margin, 0
                    )
                )
                last_row = np.ceil(
                    min(
                        np.max(ind_rows_sensor) + classif_interpolator_margin,
                        sensor_classif_image.height,
                    )
                )
                first_col = np.floor(
                    max(
                        np.min(ind_cols_sensor) - classif_interpolator_margin, 0
                    )
                )
                last_col = np.ceil(
                    min(
                        np.max(ind_cols_sensor) + classif_interpolator_margin,
                        sensor_classif_image.width,
                    )
                )

                rio_window = rio.windows.Window.from_slices(
                    (first_row, last_row),
                    (first_col, last_col),
                )

                sensor_points = (
                    np.arange(first_row, last_row),
                    np.arange(first_col, last_col),
                )

                for band in range(sensor_classif_image.count):
                    # rio band convention
                    sensor_data = sensor_classif_image.read(
                        band + 1, window=rio_window, out_dtype=np.uint8
                    )

                    filled_classif[band, :] = np.logical_or(
                        filled_classif[band, :],
                        interpolate.interpn(
                            sensor_points,
                            sensor_data,
                            (ind_rows_sensor, ind_cols_sensor),
                            bounds_error=False,
                            method=classif_interpolator,
                        ),
                    )

    return output_not_interpolated_mask, all_values


def compute_sensor_bounds(sensor_inputs, geom_plugin, output_epsg):
    """
    Compute bounds of each input sensor that have an associated color or
    classification image

    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param geom_plugin: geometry plugin used for inverse locations
    :type geom_plugin: AbstractGeometry
    :param output_epsg: epsg of the output polygons
    :type output_epsg: int

    :return: a dictionary containing a Polygon in output geometry for each
        valid input sensor
    """

    sensor_bounds = {}

    for sensor_name, sensor in sensor_inputs.items():
        reference_sensor_image = sensor.get(
            "color", sensor.get("classification", None)
        )
        # no data for this sensor, no need to compute its bounds
        if reference_sensor_image is None:
            continue

        u_l, u_r, l_r, l_l = geom_plugin.image_envelope(
            reference_sensor_image, sensor["geomodel"]
        )

        poly_geo = Polygon([u_l, u_r, l_r, l_l, u_l])

        sensor_bounds[sensor_name] = polygon_projection(
            poly_geo, 4326, output_epsg
        )

    return sensor_bounds


def filter_sensor_inputs(sensor_inputs, sensor_bounds, ground_polygon):
    """
    Filter input sensors by comparing their bounds to a reference Polygon

    :param sensor_inputs: dictionary containing paths to input images and models
    :type sensor_inputs: dict
    :param sensor_bounds: dictionary containing bounds of input sensors
    :type sensor_bounds: dict
    :param ground_polygon: reference polygon, in ground geometry
    :type ground_polygon: Polygon

    :return: a fitlered version of sensor_inputs
    """

    filtered_sensor_inputs = sensor_inputs.copy()

    for sensor_name, bound in sensor_bounds.items():
        if not bound.intersects(ground_polygon):
            del filtered_sensor_inputs[sensor_name]

    return filtered_sensor_inputs

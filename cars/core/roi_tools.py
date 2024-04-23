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
This module contains tools for ROI
"""

import logging
import os
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape

# CARS imports
from cars.core import inputs, projection


def generate_roi_poly_from_inputs(roi):
    """
    Generate ROI polygon from roi inputs

    :param roi: roi file path or Geojson
    :type roi: str Or GeoJson Dict

    :return polygon, polygon epsg
    :rtype: Tuple(Shapely Polygon, int)

    """

    roi_poly, roi_epsg = None, None
    # Transform ROI if from shapefile
    if isinstance(roi, str):
        # Transform file to shapely polygon

        roi_poly, roi_epsg = parse_roi_file(roi)
    elif isinstance(roi, dict):
        # Transform geosjon to shapely polygon
        roi_poly, roi_epsg = geojson_to_shapely(roi)

    # Resample polygon to a 100m resolution
    if roi_poly is not None:
        roi_poly = resample_polygon(roi_poly, roi_epsg, resolution=100)

    return roi_poly, roi_epsg


def resample_polygon(roi_poly, roi_epsg, resolution=100):
    """
    Resample input polygon to given resolution.
    No interpolation is applied.

    :param roi_poly: input polygon or multipolygon
    :type roi_poly: Shapely Polygon or MultiPolygon
    :param roi_epsg: roi epsg
    :type roi_epsg: int
    :param resolution: resolution in meter to resample to
    :type resolution: float

    :return: resampled polygon
    :rtype: Shapely Polygon or MultiPolygon

    """
    epsg_meter = 4978

    if roi_poly.geom_type == "Polygon":
        list_poly = [roi_poly]
    elif roi_poly.geom_type == "MultiPolygon":
        list_poly = list(roi_poly.geoms)
    else:
        raise TypeError(
            "{} type is not supported for ROI".format(roi_poly.geom_type)
        )

    new_list_poly = []
    for poly in list_poly:
        new_list_points = []

        points = poly.boundary.coords

        linestrings = [
            LineString(points[k : k + 2]) for k in range(len(points) - 1)
        ]
        for line in linestrings:
            # Get distance in meter of line
            first, last = line.coords[0], line.coords[1]
            in_cloud = np.array([[first[0], first[1]], [last[0], last[1]]])
            out_cloud = projection.points_cloud_conversion(
                in_cloud, roi_epsg, epsg_meter
            )
            new_first = Point(out_cloud[0, 0], out_cloud[0, 1])
            new_last = Point(out_cloud[1, 0], out_cloud[1, 1])
            line_distance = new_first.distance(new_last)

            # Compute number of point to generate
            nb_points = int(line_distance / resolution) + 1

            # Generate new points
            for ind in list(np.linspace(0, 1, nb_points, endpoint=True)):
                new_list_points.append(line.interpolate(ind, normalized=True))

        new_list_poly.append(Polygon(new_list_points))

    if roi_poly.geom_type == "Polygon":
        return new_list_poly[0]
    return MultiPolygon(new_list_poly)


def geojson_to_shapely(geojson_dict: dict):
    """
    Transform Geojson dict to Shapely polygon

    :param geojson_dict: geojson
    :type geojson_dict: dict

    :return: shapely polygon
    :rtype: Shapely Polygon

    """

    features = geojson_dict["features"]
    if len(features) > 1:
        logging.info(
            "Multi features files are not supported, "
            "the first feature of input geojson will be used"
        )

    roi_poly = shape(features[0]["geometry"]).buffer(0)

    roi_epsg = 4326

    if "crs" in geojson_dict:
        if "properties" in geojson_dict["crs"]:
            if "name" in geojson_dict["crs"]["properties"]:
                geo_json_epsg = geojson_dict["crs"]["properties"]["name"]
                # format: EPSG:4326
                if "EPSG:" not in geo_json_epsg:
                    logging.error("ROI EPSG could not be read, wrong format")
                else:
                    roi_epsg = int(geo_json_epsg.replace("EPSG:", ""))

    return roi_poly, roi_epsg


def parse_roi_file(arg_roi_file: str) -> Tuple[List[float], int]:
    """
    Parse ROI file argument and generate bounding box


    :param arg_roi_file : ROI file argument
    :return: ROI Polygon + ROI epsg
    :rtype: Shapely polygon, int
    """

    # Declare output
    roi_poly, roi_epsg = None, None

    _, extension = os.path.splitext(arg_roi_file)

    # test file existence
    if not os.path.exists(arg_roi_file):
        logging.error("File {} does not exist".format(arg_roi_file))
    else:
        # if it is a vector file
        if extension in [".gpkg", ".shp", ".kml"]:
            roi_poly, roi_epsg = inputs.read_vector(arg_roi_file)

        else:
            logging.error(
                "ROI file {} has an unsupported format".format(arg_roi_file)
            )
            raise AttributeError(
                "ROI file {} has an unsupported format".format(arg_roi_file)
            )

    return roi_poly, roi_epsg


def bounds_to_poly(bounds):
    """
    Convert bounds to polygon

    :param bounds: bounds: [xmin, ymin, xmax, ymax]
    :type: bounds: list

    :return polygon
    """

    poly = Polygon(
        [
            [bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]],
            [bounds[0], bounds[1]],
        ]
    )
    return poly

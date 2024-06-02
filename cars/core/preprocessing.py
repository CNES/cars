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
Preprocessing contains function used in pipelines
"""
# pylint: disable=too-many-lines

# Standard imports
from __future__ import print_function

import logging
import math
import os

import numpy as np
import utm
from pyproj import CRS
from shapely.geometry import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.applications.grid_generation import grids

# CARS imports
from cars.core import inputs, projection, tiling
from cars.core.utils import safe_makedirs
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)

PREPROCESSING_TAG = "pair_preprocessing"
LEFT_ENVELOPE_TAG = "left_envelope"
RIGHT_ENVELOPE_TAG = "right_envelope"
ENVELOPES_INTERSECTION_TAG = "envelopes_intersection"
ENVELOPES_INTERSECTION_BB_TAG = "envelopes_intersection_bounding_box"


def get_utm_zone_as_epsg_code(lon, lat):
    """
    Returns the EPSG code of the UTM zone where the lat, lon point falls in

    :param lon: longitude of the point
    :type lon: float
    :param lat: latitude of the point
    :type lat: float
    :return: The EPSG code corresponding to the UTM zone
    :rtype: int
    """

    zone = utm.from_latlon(lat, lon)[2]

    north_south = 600 if lat >= 0 else 700
    return 32000 + north_south + zone


def compute_terrain_bbox(  # noqa: 751
    sensor_image_left,
    sensor_image_right,
    epipolar_image_left,
    grid_left,
    grid_right,
    epsg,
    geometry_plugin,
    disp_min=-10,
    disp_max=10,
    resolution=0.5,
    roi_poly=None,
    pair_key="PAIR_0",
    pair_folder=None,
    orchestrator=None,
    check_inputs=False,
):
    """
    Compute terrain bounding box of current pair


    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: int
    :param geoid: geoid path
    :type geoid: str
    :param sensor_image_left: left image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_left: dict
    :param sensor_image_right: right image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_right: dict
    :param grid_left: left grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", \
                "epipolar_origin_x", "epipolar_origin_y", \
                "epipolar_spacing_x", "epipolar_spacing", \
                "disp_to_alt_ratio",
    :type grid_left: CarsDataset
    :param grid_right: right grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", "epipolar_size_y", \
                "epipolar_origin_x", "epipolar_origin_y", \
                "epipolar_spacing_x", "epipolar_spacing", \
                "disp_to_alt_ratio",
    :type grid_right: CarsDataset
    :param epsg: epsg to use
    :type epsg: str
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int
    :param resolution: resolution
    :type resolution: float
    :param roi_poly: roi polygon
    :type roi_poly: Polygon
    :param pair_key: pair key id
    :type pair_key: str
    :param pair_folder: pair folder to save data to
    :type pair_folder: str
    :param orchestrator: orchestrator
    :type orchestrator: Orchestrator
    :param check_inputs: true if user wants to check inputs
    :type check_inputs: bool

    :return: former post prepare configuration
    :rtype: dict

    """

    # Default orchestrator
    if orchestrator is None:
        # Create default sequential orchestrator for current application
        # be awere, no out_json will be shared between orchestrators
        # No files saved
        orchestrator = ocht.Orchestrator(
            orchestrator_conf={"mode": "sequential"}
        )

    if pair_folder is None:
        pair_folder = os.path.join(orchestrator.out_dir, "tmp")
        safe_makedirs(pair_folder)

    out_dir = pair_folder

    # Check that the envelopes intersect one another
    logging.info("Computing images envelopes and their intersection")
    shp1 = os.path.join(out_dir, "left_envelope.shp")
    shp2 = os.path.join(out_dir, "right_envelope.shp")
    out_envelopes_intersection = os.path.join(
        out_dir, "envelopes_intersection.gpkg"
    )

    sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
    sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
    geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
    geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

    inter_poly, (
        inter_xmin,
        inter_ymin,
        inter_xmax,
        inter_ymax,
    ) = projection.ground_intersection_envelopes(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        geometry_plugin,
        shp1,
        shp2,
        out_envelopes_intersection,
    )

    # update out_json
    updating_dict = {
        PREPROCESSING_TAG: {
            pair_key: {
                LEFT_ENVELOPE_TAG: shp1,
                RIGHT_ENVELOPE_TAG: shp2,
                ENVELOPES_INTERSECTION_TAG: out_envelopes_intersection,
                ENVELOPES_INTERSECTION_BB_TAG: [
                    inter_xmin,
                    inter_ymin,
                    inter_xmax,
                    inter_ymax,
                ],
            }
        }
    }

    orchestrator.update_out_info(updating_dict)

    if check_inputs:
        logging.info("Checking DEM coverage")
        _, epsg1 = inputs.read_vector(shp1)
        __, dem_coverage = projection.compute_dem_intersection_with_poly(
            geometry_plugin.dem, inter_poly, epsg1
        )

        if dem_coverage < 100.0:
            logging.warning(
                "The input DEM covers {}% of the useful zone".format(
                    int(dem_coverage)
                )
            )

    # Get largest epipolar regions from configuration file
    largest_epipolar_region = [
        0,
        0,
        grid_left.attributes["epipolar_size_x"],
        grid_left.attributes["epipolar_size_y"],
    ]

    # Numpy array with corners of largest epipolar region.
    # Order does not matter here,
    # since it will be passed to grids.compute_epipolar_grid_min_max
    corners = np.array(
        [
            [
                [largest_epipolar_region[0], largest_epipolar_region[1]],
                [largest_epipolar_region[0], largest_epipolar_region[3]],
            ],
            [
                [largest_epipolar_region[2], largest_epipolar_region[3]],
                [largest_epipolar_region[2], largest_epipolar_region[1]],
            ],
        ],
        dtype=np.float64,
    )

    # Compute terrain min and max again, this time using estimated epsg code
    terrain_dispmin, terrain_dispmax = grids.compute_epipolar_grid_min_max(
        geometry_plugin,
        corners,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        epsg,
        disp_min,
        disp_max,
    )

    # Compute bounds from epipolar image corners and dispmin/dispmax
    terrain_bounds = np.stack((terrain_dispmin, terrain_dispmax), axis=0)
    terrain_min = np.amin(terrain_bounds, axis=(0, 1))
    terrain_max = np.amax(terrain_bounds, axis=(0, 1))

    terrain_area = (terrain_max[0] - terrain_min[0]) * (
        terrain_max[1] - terrain_min[1]
    )

    logging.info(
        "Terrain area covered: {} square meters (or square degrees)".format(
            terrain_area
        )
    )

    # Retrieve bounding box of the ground intersection of the envelopes
    inter_poly, inter_epsg = inputs.read_vector(out_envelopes_intersection)

    # Project polygon if epsg is different
    if epsg != inter_epsg:
        inter_poly = projection.polygon_projection(inter_poly, inter_epsg, epsg)

    (inter_xmin, inter_ymin, inter_xmax, inter_ymax) = inter_poly.bounds

    # Align bounding box to integer resolution steps
    xmin, ymin, xmax, ymax = tiling.snap_to_grid(
        inter_xmin, inter_ymin, inter_xmax, inter_ymax, resolution
    )

    logging.info(
        "Terrain bounding box : [{}, {}] x [{}, {}]".format(
            xmin, xmax, ymin, ymax
        )
    )

    terrain_bounding_box = [xmin, ymin, xmax, ymax]

    # Check if roi given by user intersects with current terrain region
    if roi_poly is not None:
        if not roi_poly.intersects(inter_poly):
            logging.warning(
                "The pair composed of {} and {} "
                "does not intersect the requested ROI".format(
                    sensor_image_left[sens_cst.INPUT_IMG],
                    sensor_image_right[sens_cst.INPUT_IMG],
                )
            )

    # Get number of epipolar tiles that are previously used
    nb_epipolar_tiles = (
        epipolar_image_left.shape[0] * epipolar_image_left.shape[1]
    )

    # Compute average epipolar tile width
    epipolar_average_tile_width = math.sqrt(terrain_area / nb_epipolar_tiles)

    return (terrain_bounding_box, epipolar_average_tile_width)


def compute_roi_poly(input_roi_poly, input_roi_epsg, epsg):
    """
    Compute roi polygon from input roi

    :param input_roi_poly: roi polygon
    :type input_roi_poly:  shapely Polygon
    :param input_roi_epsg: epsg of roi
    :type input_roi_epsg: str
    :param epsg: epsg to use
    :type epsg: str

    :return: polygon of roi with right epsg
    :rtype: Polygon

    """
    roi_poly = input_roi_poly

    if input_roi_poly is not None:
        if input_roi_epsg != epsg:
            roi_poly = projection.polygon_projection(
                roi_poly, input_roi_epsg, epsg
            )

    return roi_poly


def compute_epsg(
    sensor_image_left,
    sensor_image_right,
    grid_left,
    grid_right,
    geometry_plugin,
    disp_min=-10,
    disp_max=10,
):
    """
    Compute epsg to use

    :param sensor_image_left: left image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_left: dict
    :param sensor_image_right: right image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_right: dict
    :param grid_left: left grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", \
                "epipolar_origin_x", "epipolar_origin_y", \
                "epipolar_spacing_x", "epipolar_spacing", \
                "disp_to_alt_ratio",
    :type grid_left: CarsDataset
    :param grid_right: right grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", \
                "epipolar_origin_x", "epipolar_origin_y", \
                "epipolar_spacing_x", "epipolar_spacing", \
                "disp_to_alt_ratio",
    :type grid_right: CarsDataset
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: int
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int

    :return: epsg
    :rtype: str

    """
    sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
    sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
    geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
    geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

    # Get largest epipolar regions from configuration file
    largest_epipolar_region = [
        0,
        0,
        grid_left.attributes["epipolar_size_x"],
        grid_left.attributes["epipolar_size_y"],
    ]

    # Numpy array with corners of largest epipolar region.
    # Order does not matter here,
    # since it will be passed to grids.compute_epipolar_grid_min_max
    corners = np.array(
        [
            [
                [largest_epipolar_region[0], largest_epipolar_region[1]],
                [largest_epipolar_region[0], largest_epipolar_region[3]],
            ],
            [
                [largest_epipolar_region[2], largest_epipolar_region[3]],
                [largest_epipolar_region[2], largest_epipolar_region[1]],
            ],
        ],
        dtype=np.float64,
    )

    # Compute epipolar image terrain position corners
    # for min and max disparity
    (
        terrain_dispmin,
        _,
    ) = grids.compute_epipolar_grid_min_max(
        geometry_plugin,
        corners,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        4326,
        disp_min,
        disp_max,
    )

    epsg = get_utm_zone_as_epsg_code(*np.mean(terrain_dispmin, axis=0))

    logging.info("EPSG code: {}".format(epsg))

    return epsg


def crop_terrain_bounds_with_roi(roi_poly, xmin, ymin, xmax, ymax):
    """
    Crop current terrain bounds with roi

    :param roi_poly: Polygon of ROI
    :type roi_poly: Shapely Polygon
    :param xmin: xmin
    :type xmin: float
    :param ymin: ymin
    :type ymin: float
    :param xmax: xmax
    :type xmax: float
    :param ymax: ymax
    :type ymax: float

    :return: new xmin, ymin, xmax, ymax
    :rtype: (float, float, float, float)
    """
    # terrain bounding box polygon
    terrain_poly = Polygon(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
    )

    if not roi_poly.intersects(terrain_poly):
        raise RuntimeError("None of the input data intersect the requested ROI")
    # Show ROI if valid (no exception raised) :
    logging.info("Setting terrain bounding box to the requested ROI")
    new_xmin, new_ymin, new_xmax, new_ymax = roi_poly.bounds

    return new_xmin, new_ymin, new_xmax, new_ymax


def compute_terrain_bounds(list_of_terrain_roi, roi_poly=None, resolution=0.5):
    """
    Compute Terrain bounds of merged pairs

    :param list_of_terrain_roi: list of terrain roi
            list of (terrain bbox, terrain epi_tile_size)
    :type list_of_terrain_roi: list
    :param roi_poly: terrain roi of given roi
    :type roi_poly: Polygon
    :param resolution: list of terrain roi
    :type resolution: float

    :return: bounds, optimal_terrain_tile_width_average

    """

    # get lists
    (
        list_terrain_bounding_box,
        list_terrain_epi_tile_width,
    ) = zip(  # noqa: B905
        *list_of_terrain_roi
    )
    list_terrain_bounding_box = list(list_terrain_bounding_box)
    list_terrain_epi_tile_width = list(list_terrain_epi_tile_width)

    xmin, ymin, xmax, ymax = tiling.union(list_terrain_bounding_box)

    if roi_poly is not None:
        (xmin, ymin, xmax, ymax) = crop_terrain_bounds_with_roi(
            roi_poly, xmin, ymin, xmax, ymax
        )

        xmin, ymin, xmax, ymax = tiling.snap_to_grid(
            xmin, ymin, xmax, ymax, resolution
        )

    logging.info(
        "Total terrain bounding box : [{}, {}] x [{}, {}]".format(
            xmin, xmax, ymin, ymax
        )
    )

    bounds = [xmin, ymin, xmax, ymax]

    # Compute optimal terrain tile width
    optimal_terrain_tile_width_average = np.nanmean(list_terrain_epi_tile_width)

    optimal_terrain_tile_width = (
        int(math.ceil(optimal_terrain_tile_width_average / resolution))
        * resolution
    )

    logging.info(
        "Optimal terrain tile size: {}x{} pixels".format(
            int(optimal_terrain_tile_width / resolution),
            int(optimal_terrain_tile_width / resolution),
        )
    )

    return bounds, optimal_terrain_tile_width


def get_conversion_factor(bounds, epsg, epsg_cloud):
    """
    Conmpute conversion factor

    :param bounds: terrain bounds
    :type bounds: list
    :param epsg: epsg of bounds
    :type epsg: int
    :param epsg_cloud: epsg of the input cloud
    :type epsg_cloud: int
    :return: conversion factor
    :rtype: float
    """

    conversion_factor = 1

    # only if epsg and epsg_cloud are different
    spatial_ref = CRS.from_epsg(epsg)
    spatial_ref_cloud = CRS.from_epsg(epsg_cloud)
    is_geographic = spatial_ref.is_geographic or spatial_ref_cloud.is_geographic
    if is_geographic and epsg != epsg_cloud:
        # Compute bounds and terrain grid
        [xmin, ymin, xmax, ymax] = bounds
        bounds_points = [
            [xmin, ymin],
            [xmax, ymax],
        ]
        bounds_points_epsg_cloud = projection.points_cloud_conversion(
            bounds_points, epsg, epsg_cloud
        )
        # Compute area in both epsg
        terrain_area_epsg = (xmax - xmin) * (ymax - ymin)
        terrain_area_epsg_cloud = (
            bounds_points_epsg_cloud[1][0] - bounds_points_epsg_cloud[0][0]
        ) * (bounds_points_epsg_cloud[1][1] - bounds_points_epsg_cloud[0][1])
        # Compute conversion factor
        conversion_factor = math.sqrt(
            terrain_area_epsg / terrain_area_epsg_cloud
        )

    return conversion_factor


def convert_optimal_tile_size_with_epsg(
    bounds, optimal_terrain_tile_width, epsg, epsg_cloud
):
    """
    Convert optimal_tile_size according to epsg.
    Only if epsg_cloud is different of the output epsg.

    :param bounds: terrain bounds
    :type bounds: list
    :param optimal_terrain_tile_width: initial optimal_terrain_tile_width
    :type optimal_terrain_tile_width: float
    :param epsg: target epsg
    :type epsg: int
    :param epsg_cloud: epsg of the input cloud
    :type epsg_cloud: int
    :return: converted optimal tile size
    :rtype: float
    """

    # Convert optimal terrain tile width
    conversion_factor = get_conversion_factor(bounds, epsg, epsg_cloud)
    optimal_terrain_tile_width *= conversion_factor
    return optimal_terrain_tile_width


def compute_epipolar_roi(
    terrain_roi_poly,
    terrain_roi_epsg,
    geometry_plugin,
    sensor_image_left,
    sensor_image_right,
    grid_left,
    grid_right,
    output_path,
    disp_min=0,
    disp_max=0,
):
    """
    Compute epipolar roi to use

    :param terrain_roi_poly: terrain  roi polygon
    :param terrain_roi_epsg: terrain  roi epsg
    :param geometry_plugin: geometry plugin to use
    :param epsg: epsg
    :param disp_min: minimum disparity
    :param disp_max: maximum disparity

    :return: epipolar region to use, with tile_size a sample
    """

    if terrain_roi_poly is None:
        return None

    roi_bbox = terrain_roi_poly.bounds

    pair_folder = os.path.join(output_path, "tmp")
    safe_makedirs(pair_folder)

    sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
    sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
    geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
    geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

    epipolar_roi = grids.terrain_region_to_epipolar(
        roi_bbox,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        grid_left,
        grid_right,
        geometry_plugin,
        epsg=terrain_roi_epsg,
        disp_min=disp_min,
        disp_max=disp_max,
        tile_size=100,
        epipolar_size_x=grid_left.attributes["epipolar_size_x"],
        epipolar_size_y=grid_left.attributes["epipolar_size_y"],
    )

    return epipolar_roi

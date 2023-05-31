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
from shapely.geometry import Polygon

import cars.conf.input_parameters as in_params
import cars.orchestrator.orchestrator as ocht
from cars.applications.grid_generation import grids

# CARS imports
from cars.core import former_confs_utils, inputs, projection, tiling
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
    srtm_dir,
    default_alt,
    geoid,
    sensor_image_left,
    sensor_image_right,
    epipolar_image_left,
    grid_left,
    grid_right,
    epsg,
    geometry_loader_to_use,
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
    :param geometry_loader_to_use: geometry loader to use
    :type geometry_loader_to_use: str
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

    input_configuration = create_former_cars_conf(
        sensor_image_left,
        sensor_image_right,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
    )

    inter_poly, (
        inter_xmin,
        inter_ymin,
        inter_xmax,
        inter_ymax,
    ) = projection.ground_intersection_envelopes(
        input_configuration,
        geometry_loader_to_use,
        shp1,
        shp2,
        out_envelopes_intersection,
        geoid=geoid,
        dem_dir=srtm_dir,
        default_alt=default_alt,
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
            srtm_dir, inter_poly, epsg1
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

    # Create fake configuration  TODO remove
    configuration = create_former_cars_post_prepare_configuration(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        pair_folder,
        uncorrected_grid_right=None,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
    )

    # Compute terrain min and max again, this time using estimated epsg code
    terrain_dispmin, terrain_dispmax = grids.compute_epipolar_grid_min_max(
        geometry_loader_to_use, corners, epsg, configuration, disp_min, disp_max
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
    geometry_loader_to_use,
    orchestrator=None,
    pair_folder=None,
    srtm_dir=None,
    default_alt=None,
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
    :param geometry_loader_to_use: geometry loader to use
    :type geometry_loader_to_use: str
    :param orchestrator: orchestrator
    :type orchestrator: Orchestrator
    :param pair_folder: pair folder to save data to
    :type pair_folder: str
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

    if pair_folder is None:
        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        pair_folder = os.path.join(orchestrator.out_dir, "pair_0")

    # get UTM zone with the middle point of terrain_min if epsg is None

    # Create fake configuration  TODO remove
    configuration = create_former_cars_post_prepare_configuration(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        pair_folder,
        uncorrected_grid_right=None,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
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

    # Compute epipolar image terrain position corners
    # for min and max disparity
    (
        terrain_dispmin,
        _,
    ) = grids.compute_epipolar_grid_min_max(
        geometry_loader_to_use, corners, 4326, configuration, disp_min, disp_max
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
    optimal_terrain_tile_width_average = np.mean(list_terrain_epi_tile_width)

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


def create_former_cars_post_prepare_configuration(
    sensor_image_left,
    sensor_image_right,
    grid_left,
    grid_right,
    pair_folder,
    uncorrected_grid_right=None,
    srtm_dir=None,
    default_alt=0,
    disp_min=None,
    disp_max=None,
):
    """
    Create post prepare configuration used in former version of CARS

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
    :param pair_folder: pair folder to save data to
    :type pair_folder: str
    :param uncorrected_grid_right: uncorrected right grid
    :type uncorrected_grid_right: CarsDataset
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: int
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int

    :return: former post prepare configuration
    :rtype: dict

    """

    # TODO remove with refactoring low level configuration

    input_configuration = create_former_cars_conf(
        sensor_image_left,
        sensor_image_right,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
    )

    configuration = {}
    # TODO change it, modify geometry loader inputs
    configuration[in_params.INPUT_SECTION_TAG] = input_configuration
    # Save grids
    safe_makedirs(os.path.join(pair_folder, "tmp"))
    grid_origin = grid_left.attributes["grid_origin"]
    grid_spacing = grid_left.attributes["grid_spacing"]
    left_grid_path = grids.get_new_path(
        os.path.join(pair_folder, "tmp", "left_epi_grid.tif")
    )
    grids.write_grid(grid_left[0, 0], left_grid_path, grid_origin, grid_spacing)

    right_grid_path = grids.get_new_path(
        os.path.join(pair_folder, "tmp", "corrected_right_epi_grid.tif")
    )
    grids.write_grid(
        grid_right[0, 0], right_grid_path, grid_origin, grid_spacing
    )

    uncorrected_right_grid_path = grids.get_new_path(
        os.path.join(pair_folder, "tmp", "right_epi_grid_uncorrected.tif")
    )
    if uncorrected_grid_right is not None:
        grids.write_grid(
            uncorrected_grid_right[0, 0],
            uncorrected_right_grid_path,
            grid_origin,
            grid_spacing,
        )

    # add to configuration
    configuration[former_confs_utils.PREPROCESSING_SECTION_TAG] = {}

    output_conf = {}

    output_conf[former_confs_utils.MINIMUM_DISPARITY_TAG] = disp_min
    output_conf[former_confs_utils.MAXIMUM_DISPARITY_TAG] = disp_max

    output_conf[former_confs_utils.LEFT_EPIPOLAR_GRID_TAG] = left_grid_path
    output_conf[former_confs_utils.RIGHT_EPIPOLAR_GRID_TAG] = right_grid_path
    if uncorrected_grid_right is not None:
        output_conf[
            former_confs_utils.RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG
        ] = uncorrected_right_grid_path
    output_conf[former_confs_utils.EPIPOLAR_SIZE_X_TAG] = grid_left.attributes[
        "epipolar_size_x"
    ]
    output_conf[former_confs_utils.EPIPOLAR_SIZE_Y_TAG] = grid_left.attributes[
        "epipolar_size_y"
    ]

    output_conf[
        former_confs_utils.DISP_TO_ALT_RATIO_TAG
    ] = grid_left.attributes["disp_to_alt_ratio"]

    if disp_min is not None:
        output_conf[former_confs_utils.MINIMUM_DISPARITY_TAG] = disp_min

    if disp_max is not None:
        output_conf[former_confs_utils.MAXIMUM_DISPARITY_TAG] = disp_max

    configuration[former_confs_utils.PREPROCESSING_SECTION_TAG][
        former_confs_utils.PREPROCESSING_OUTPUT_SECTION_TAG
    ] = output_conf

    return configuration


def create_former_cars_conf(  # noqa: C901
    sensor_image_left, sensor_image_right, srtm_dir=None, default_alt=0
):
    """
    Create input configuration used in former version of CARS

    :param sensor_image_left: left image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_left: dict
    :param sensor_image_right: right image
           Dict Must contain keys : "image", "color", "geomodel",
           "no_data", "mask". Paths must be absolutes
    :type sensor_image_right: dict
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: int

    :return: former input configuration
    :rtype: dict

    """

    # TODO remove this function, update geometry loader
    conf = {
        "img1": sensor_image_left[sens_cst.INPUT_IMG],
        "img2": sensor_image_right[sens_cst.INPUT_IMG],
    }

    if "geomodel" in sensor_image_left:
        if sensor_image_left["geomodel"] is not None:
            conf["model1"] = sensor_image_left["geomodel"]

    if "geomodel" in sensor_image_right:
        if sensor_image_right["geomodel"] is not None:
            conf["model2"] = sensor_image_right["geomodel"]

    if "geomodel_type" in sensor_image_left:
        if sensor_image_left["geomodel"] is not None:
            conf["model_type1"] = sensor_image_left["geomodel_type"]

    if "geomodel_type" in sensor_image_right:
        if sensor_image_right["geomodel"] is not None:
            conf["model_type2"] = sensor_image_right["geomodel_type"]

    if "geomodel_filters" in sensor_image_left:
        if sensor_image_left["geomodel_filters"] is not None:
            conf["model1_filters"] = sensor_image_left["geomodel_filters"]

    if "geomodel_filters" in sensor_image_right:
        if sensor_image_right["geomodel_filters"] is not None:
            conf["model2_filters"] = sensor_image_right["geomodel_filters"]

    if "mask" in sensor_image_left:
        if sensor_image_left["mask"] is not None:
            conf["mask1"] = sensor_image_left["mask"]

    if "mask" in sensor_image_right:
        if sensor_image_left["mask"] is not None:
            conf["mask1"] = sensor_image_left["mask"]

    if "color" in sensor_image_left:
        if sensor_image_left["color"] is not None:
            conf["color1"] = sensor_image_left["color"]

    conf["srtm_dir"] = srtm_dir
    conf["default_alt"] = default_alt

    return conf


def compute_epipolar_roi(
    terrain_roi_poly,
    terrain_roi_epsg,
    geometry_loader,
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
    :param geometry_loader: geometry loader to use
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

    conf = create_former_cars_post_prepare_configuration(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        pair_folder,
        uncorrected_grid_right=None,
        srtm_dir=None,
        default_alt=0,
        disp_min=disp_min,
        disp_max=disp_max,
    )

    epipolar_roi = grids.terrain_region_to_epipolar(
        roi_bbox,
        conf,
        geometry_loader,
        epsg=terrain_roi_epsg,
        disp_min=disp_min,
        disp_max=disp_max,
        tile_size=100,
        epipolar_size_x=grid_left.attributes["epipolar_size_x"],
        epipolar_size_y=grid_left.attributes["epipolar_size_y"],
    )

    return epipolar_roi

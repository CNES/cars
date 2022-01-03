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
Main CARS prepare pipeline module:
contains functions associated to prepare CARS sub-command
TODO: refactor in several files and remove too-many-lines
"""
# pylint: disable=too-many-lines

# Standard imports
from __future__ import absolute_import, print_function

import errno
import logging
import math
import os
import pickle

# Third party imports
import dask
import numpy as np
import rasterio as rio
from dask.distributed import as_completed
from json_checker import CheckerError
from tqdm import tqdm

# CARS imports
from cars import __version__
from cars.cluster import start_cluster, stop_cluster
from cars.conf import input_parameters as in_params
from cars.conf import (
    json_schema,
    log_conf,
    mask_classes,
    output_prepare,
    static_conf,
)
from cars.core import constants as cst
from cars.core import inputs, projection, tiling
from cars.externals import otb_pipelines
from cars.pipelines.wrappers import matching_wrapper
from cars.steps import devib, rasterization, triangulation
from cars.steps.epi_rectif import grids
from cars.steps.matching import sparse_matching


def run(  # noqa: C901
    in_json: in_params.InputConfigurationType,
    out_dir: str,
    epi_step: int = 30,
    region_size: int = 500,
    disparity_margin: float = 0.02,
    epipolar_error_upper_bound: float = 10.0,
    epipolar_error_maximum_bias: float = 0.0,
    elevation_delta_lower_bound: float = -1000.0,
    elevation_delta_upper_bound: float = 1000.0,
    mode: str = "local_dask",
    nb_workers: int = 4,
    walltime: str = "00:59:00",
    check_inputs: bool = False,
):
    """
    Main function of the prepare step subcommand

    This function will perform the following steps:

    1. Compute stereo-rectification grids for the input pair
    2. Compute all possible sift matches in epipolar geometry
    3. Derive an optimal disparity range to explore from the matches
    4. Derive a bilinear correction model of the stereo-rectification grid
        for right image in order to minimize epipolar error
    5. Apply correction to right grid
    6. Export left and corrected right grid

    :param in_json:  dictionary describing input data
        (see README.md for format)
    :param out_dir: Directory where all outputs will be written,
        including a content.json file describing its content
    :param epi_step: Step of the epipolar grid to compute
        (in pixels in epipolar geometry)
    :param region_size: Size of regions used for sift matching
    :param disparity_margin: Percent of the disparity range width
        to add at each end as security margin
    :param epipolar_error_upper_bound: Upper bound
        of expected epipolar error (in pixels)
    :param epipolar_error_maximum_bias: Maximum bias
        for epipolar error (in pixels)
    :param elevation_delta_lower_bound: Lower bound for elevation delta
        with respect to initial DSM (in meters)
    :param elevation_delta_upper_bound: Upper bound for elevation delta
        with respect to initial DSM (in meters)
    :param mode: Parallelization mode
    :param nb_workers: Number of dask workers to use for the sift matching step
    :param walltime: Walltime of the dask workers
    :param check_inputs: activation of the inputs consistency checking
    """
    out_dir = os.path.abspath(out_dir)

    # Ensure that outdir exists
    try:
        os.makedirs(out_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise

    log_conf.add_log_file(out_dir, "prepare")

    if not check_inputs:
        logging.info(
            "The inputs consistency will not be checked. "
            "To enable the inputs checking, add --check_inputs "
            "to your command line"
        )

    # Check configuration dict
    config = inputs.check_json(in_json, json_schema.input_conf_schema())

    # Retrieve static parameters (sift and low res dsm)
    static_params = static_conf.get_cfg()

    # Initialize output json dict
    # (use local variables to keep indentation and line not too long)
    epi_step_tag = output_prepare.EPI_STEP_TAG
    disp_margin_tag = output_prepare.DISPARITY_MARGIN_TAG
    epi_error_up_bound_tag = output_prepare.EPIPOLAR_ERROR_UPPER_BOUND_TAG
    epi_error_max_bias_tag = output_prepare.EPIPOLAR_ERROR_MAXIMUM_BIAS_TAG
    elev_delta_low_bound_tag = output_prepare.ELEVATION_DELTA_LOWER_BOUND_TAG
    elev_delta_up_bound_tag = output_prepare.ELEVATION_DELTA_UPPER_BOUND_TAG
    out_json = {
        in_params.INPUT_SECTION_TAG: config,
        output_prepare.PREPROCESSING_SECTION_TAG: {
            output_prepare.PREPROCESSING_VERSION_TAG: __version__,
            output_prepare.PREPROCESSING_PARAMETERS_SECTION_TAG: {
                epi_step_tag: epi_step,
                disp_margin_tag: disparity_margin,
                epi_error_up_bound_tag: epipolar_error_upper_bound,
                epi_error_max_bias_tag: epipolar_error_maximum_bias,
                elev_delta_low_bound_tag: elevation_delta_lower_bound,
                elev_delta_up_bound_tag: elevation_delta_upper_bound,
            },
            in_params.STATIC_PARAMS_TAG: {
                static_conf.prepare_tag: static_params[static_conf.prepare_tag],
                static_conf.loaders_tag: static_params[static_conf.loaders_tag],
                static_conf.geoid_path_tag: static_conf.get_geoid_path(),
            },
            output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG: {},
        },
    }

    # Read input parameters
    # TODO Refacto conf : why not a config.get() for these two here ?
    # TODO looks like we only get in_params from the config file,
    #      so maybe a single conf.read_config() or smth like that ?
    img1 = config[in_params.IMG1_TAG]
    img2 = config[in_params.IMG2_TAG]

    srtm_dir = config.get(in_params.SRTM_DIR_TAG, None)
    nodata1 = config.get(in_params.NODATA1_TAG, None)
    nodata2 = config.get(in_params.NODATA2_TAG, None)
    mask1 = config.get(in_params.MASK1_TAG, None)
    mask2 = config.get(in_params.MASK2_TAG, None)
    mask1_classes = config.get(in_params.MASK1_CLASSES_TAG, None)
    mask2_classes = config.get(in_params.MASK2_CLASSES_TAG, None)
    default_alt = config.get(in_params.DEFAULT_ALT_TAG, 0)

    # retrieve masks classes usages
    classes_usage = {}
    if mask1_classes is not None:
        mask1_classes_dict = mask_classes.read_mask_classes(mask1_classes)
        classes_usage[
            output_prepare.MASK1_IGNORED_BY_SIFT_MATCHING_TAG
        ] = mask1_classes_dict.get(
            mask_classes.ignored_by_sift_matching_tag, None
        )

    if mask2_classes is not None:
        mask2_classes_dict = mask_classes.read_mask_classes(mask2_classes)
        classes_usage[
            output_prepare.MASK2_IGNORED_BY_SIFT_MATCHING_TAG
        ] = mask2_classes_dict.get(
            mask_classes.ignored_by_sift_matching_tag, None
        )

    if mask1_classes is not None or mask2_classes is not None:
        out_json[output_prepare.PREPROCESSING_SECTION_TAG][
            output_prepare.PREPROCESSING_PARAMETERS_SECTION_TAG
        ][output_prepare.PREPARE_MASK_CLASSES_USAGE_TAG] = classes_usage

    # log information considering reference altitudes used
    if srtm_dir is not None:
        srtm_tiles = os.listdir(srtm_dir)
        if len(srtm_tiles) == 0:
            logging.warning(
                "SRTM directory is empty, "
                "the default altitude will be used as reference altitude."
            )
        else:
            logging.info(
                "Indicated SRTM tiles valid regions "
                "will be used as reference altitudes "
                "(the default altitude is used "
                "for undefined regions of the SRTM)"
            )
    else:
        logging.info("The default altitude will be used as reference altitude.")

    if check_inputs:
        logging.info("Checking inputs consistency")

        if (
            inputs.rasterio_get_nb_bands(img1) != 1
            or inputs.rasterio_get_nb_bands(img2) != 1
        ):
            raise Exception(
                "{} and {} are not mono-band images".format(img1, img2)
            )

        if mask1 is not None:
            if inputs.rasterio_get_size(img1) != inputs.rasterio_get_size(
                mask1
            ):
                raise Exception(
                    "The image {} and the mask {} "
                    "do not have the same size".format(img1, mask1)
                )

        if mask2 is not None:
            if inputs.rasterio_get_size(img2) != inputs.rasterio_get_size(
                mask2
            ):
                raise Exception(
                    "The image {} and the mask {} "
                    "do not have the same size".format(img2, mask2)
                )

        with rio.open(img1) as img1_reader:
            trans = img1_reader.transform
            if trans.e < 0:
                logging.warning(
                    "{} seems to have an incoherent pixel size. "
                    "Input images has to be in sensor geometry.".format(img1)
                )

        with rio.open(img2) as img2_reader:
            trans = img2_reader.transform
            if trans.e < 0:
                logging.warning(
                    "{} seems to have an incoherent pixel size. "
                    "Input images has to be in sensor geometry.".format(img2)
                )

    # Check geometric models consistency
    if not static_conf.get_geometry_loader().check_products_consistency(config):
        raise Exception("Problem while reading the geometric models")

    # Check that the envelopes intersect one another
    logging.info("Computing images envelopes and their intersection")
    shp1 = os.path.join(out_dir, "left_envelope.shp")
    shp2 = os.path.join(out_dir, "right_envelope.shp")
    out_envelopes_intersection = os.path.join(
        out_dir, "envelopes_intersection.gpkg"
    )

    inter_poly, (
        inter_xmin,
        inter_ymin,
        inter_xmax,
        inter_ymax,
    ) = projection.ground_intersection_envelopes(
        config,
        shp1,
        shp2,
        out_envelopes_intersection,
        dem_dir=srtm_dir,
        default_alt=default_alt,
    )

    conf_out_dict = out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ]
    conf_out_dict[output_prepare.LEFT_ENVELOPE_TAG] = shp1
    conf_out_dict[output_prepare.RIGHT_ENVELOPE_TAG] = shp2
    conf_out_dict[
        output_prepare.ENVELOPES_INTERSECTION_TAG
    ] = out_envelopes_intersection
    conf_out_dict[output_prepare.ENVELOPES_INTERSECTION_BB_TAG] = [
        inter_xmin,
        inter_ymin,
        inter_xmax,
        inter_ymax,
    ]

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

    # Generate rectification grids
    (
        grid1,
        grid2,
        grid_origin,
        grid_spacing,
        epipolar_size,
        disp_to_alt_ratio,
    ) = grids.generate_epipolar_grids(
        config,
        dem=srtm_dir,
        default_alt=default_alt,
        epipolar_step=epi_step,
        geoid=static_conf.get_geoid_path(),
    )

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_SIZE_X_TAG] = epipolar_size[0]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_SIZE_Y_TAG] = epipolar_size[1]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_ORIGIN_X_TAG] = grid_origin[0]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_ORIGIN_Y_TAG] = grid_origin[1]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_SPACING_X_TAG] = grid_spacing[0]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.EPIPOLAR_SPACING_Y_TAG] = grid_spacing[1]

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.DISP_TO_ALT_RATIO_TAG] = disp_to_alt_ratio

    logging.info(
        "Size of epipolar images: {}x{} pixels".format(
            epipolar_size[0], epipolar_size[1]
        )
    )
    logging.info(
        "Disparity to altitude factor: {} m/pixel".format(disp_to_alt_ratio)
    )

    # Get satellites angles from ground: Azimuth to north, Elevation angle
    (
        left_az,
        left_elev_angle,
        right_az,
        right_elev_angle,
        convergence_angle,
    ) = projection.get_ground_angles(config)

    logging.info(
        "Left  satellite coverture: Azimuth angle : {:.1f}°, "
        "Elevation angle: {:.1f}°".format(left_az, left_elev_angle)
    )

    logging.info(
        "Right satellite coverture: Azimuth angle : {:.1f}°, "
        "Elevation angle: {:.1f}°".format(right_az, right_elev_angle)
    )

    logging.info(
        "Stereo satellite convergence angle from ground: {:.1f}°".format(
            convergence_angle
        )
    )

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.LEFT_AZIMUTH_ANGLE_TAG] = left_az
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.LEFT_ELEVATION_ANGLE_TAG] = left_elev_angle
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.RIGHT_AZIMUTH_ANGLE_TAG] = right_az
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.RIGHT_ELEVATION_ANGLE_TAG] = right_elev_angle
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.CONVERGENCE_ANGLE_TAG] = convergence_angle

    logging.info("Sparse matching ...")

    # Compute the full range needed for sparse matching
    disp_lower_bound = elevation_delta_lower_bound / disp_to_alt_ratio
    disp_upper_bound = elevation_delta_upper_bound / disp_to_alt_ratio

    disparity_range_width = disp_upper_bound - disp_lower_bound
    logging.info(
        "Full disparity range width "
        "for sparse matching: {} pixels".format(disparity_range_width)
    )
    disparity_range_center = (
        elevation_delta_upper_bound + elevation_delta_lower_bound
    ) / (2 * disp_to_alt_ratio)

    # Compute the number of offsets to consider so as to explore the full range
    nb_splits = 1 + int(math.floor(float(disparity_range_width) / region_size))
    actual_region_size = int(
        math.ceil((region_size + disparity_range_width) / nb_splits)
    )
    actual_range = nb_splits * actual_region_size
    actual_range_start = (
        disparity_range_center - actual_range / 2 + region_size / 2
    )
    logging.info(
        "Disparity range will be explored "
        "in {} regions of size {}, starting at {} pixels".format(
            nb_splits, actual_region_size, actual_range_start
        )
    )

    regions = tiling.split(
        0, 0, epipolar_size[0], epipolar_size[1], region_size, region_size
    )

    logging.info(
        "Number of splits to process for sparse matching: {}".format(
            len(regions)
        )
    )

    # Start cluster and client
    # TODO: implement prepare mp mode to be symetric with compute_dsm.
    if mode == "mp":
        raise NotImplementedError("{} mode is not implemented yet".format(mode))
    cluster, client, _ = start_cluster(
        mode, nb_workers, walltime, out_dir, "prepare"
    )

    # Write temporary grid
    tmp1 = os.path.join(out_dir, "tmp1.tif")
    grids.write_grid(grid1, tmp1, grid_origin, grid_spacing)
    tmp2 = os.path.join(out_dir, "tmp2.tif")
    grids.write_grid(grid2, tmp2, grid_origin, grid_spacing)

    # Compute margins for right region
    margins = [
        int(
            math.floor(epipolar_error_upper_bound + epipolar_error_maximum_bias)
        ),
        int(
            math.floor(epipolar_error_upper_bound + epipolar_error_maximum_bias)
        ),
        int(
            math.floor(epipolar_error_upper_bound + epipolar_error_maximum_bias)
        ),
        int(
            math.ceil(epipolar_error_upper_bound + epipolar_error_maximum_bias)
        ),
    ]

    logging.info(
        "Margins added to right region for matching: {}".format(margins)
    )

    # Matching tasks as delayed objects
    delayed_matches = []
    for left_region in regions:
        for offset in range(nb_splits):
            offset_ = actual_range_start + offset * actual_region_size
            # Pad region to include margins for right image
            right_region = [
                left_region[0] + offset_,
                left_region[1],
                left_region[0] + offset_ + actual_region_size,
                left_region[3],
            ]

            # Pad with margin and crop to largest region
            right_region = tiling.crop(
                tiling.pad(right_region, margins),
                [0, 0, epipolar_size[0], epipolar_size[1]],
            )

            # Avoid empty regions
            if not tiling.empty(right_region):

                delayed_matches.append(
                    dask.delayed(matching_wrapper)(
                        left_region,
                        right_region,
                        img1,
                        img2,
                        tmp1,
                        tmp2,
                        mask1,
                        mask2,
                        mask1_classes,
                        mask2_classes,
                        nodata1,
                        nodata2,
                        epipolar_size[0],
                        epipolar_size[1],
                    )
                )

    # Transform delayed tasks to future
    logging.info("Submitting {} tasks to dask".format(len(delayed_matches)))
    future_matches = client.compute(delayed_matches)

    # Initialize output matches array
    matches = np.empty((0, 4))

    # Wait for all matching tasks to be completed
    for __, result in tqdm(
        as_completed(future_matches, with_results=True),
        total=len(future_matches),
        desc="Performing matching ...",
    ):
        matches = np.concatenate((matches, result))

    raw_nb_matches = matches.shape[0]

    logging.info(
        "Raw number of matches found: {} matches".format(raw_nb_matches)
    )

    # Export matches
    logging.info("Writing raw matches file")
    raw_matches_array_path = os.path.join(out_dir, "raw_matches.npy")
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.RAW_MATCHES_TAG] = raw_matches_array_path
    np.save(raw_matches_array_path, matches)

    # Filter matches that are out of margin
    if epipolar_error_maximum_bias == 0:
        epipolar_median_shift = 0
    else:
        epipolar_median_shift = np.median(matches[:, 3] - matches[:, 1])

    matches = matches[
        ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
        >= -epipolar_error_upper_bound
    ]
    matches = matches[
        ((matches[:, 3] - matches[:, 1]) - epipolar_median_shift)
        <= epipolar_error_upper_bound
    ]

    matches_discarded_message = "{} matches discarded \
        because their epipolar error is greater \
        than --epipolar_error_upper_bound = {} pix".format(
        raw_nb_matches - matches.shape[0], epipolar_error_upper_bound
    )

    if epipolar_error_maximum_bias != 0:
        matches_discarded_message += " considering a shift of {} pix".format(
            epipolar_median_shift
        )

    logging.info(matches_discarded_message)

    filtered_nb_matches = matches.shape[0]

    matches = matches[matches[:, 2] - matches[:, 0] >= disp_lower_bound]
    matches = matches[matches[:, 2] - matches[:, 0] <= disp_upper_bound]

    logging.info(
        "{} matches discarded because they fall outside of disparity range "
        "defined by --elevation_delta_lower_bound = {} m and "
        "--elevation_delta_upper_bound = {} m : [{} pix., {} pix.]".format(
            filtered_nb_matches - matches.shape[0],
            elevation_delta_lower_bound,
            elevation_delta_upper_bound,
            disp_lower_bound,
            disp_upper_bound,
        )
    )

    # Retrieve number of matches
    nb_matches = matches.shape[0]

    # Check if we have enough matches
    # TODO: we could also make it a warning and continue with uncorrected grid
    # and default disparity range
    if nb_matches < 100:
        logging.error(
            "Insufficient amount of matches found (< 100), can not safely "
            "estimate epipolar error correction and disparity range"
        )
        # stop cluster
        stop_cluster(mode, cluster, client)
        # Exit immediately
        return

    logging.info(
        "Number of matches kept for epipolar "
        "error correction: {} matches".format(nb_matches)
    )

    # Remove temporary files
    os.remove(tmp1)
    os.remove(tmp2)

    # Compute epipolar error
    epipolar_error = matches[:, 1] - matches[:, 3]
    logging.info(
        "Epipolar error before correction: mean = {:.3f} pix., "
        "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
            np.mean(epipolar_error),
            np.std(epipolar_error),
            np.max(np.fabs(epipolar_error)),
        )
    )

    # Commpute correction for right grid
    logging.info("Generating correction for right epipolar grid ...")
    corrected_right_grid, corrected_matches, __, __ = grids.correct_right_grid(
        matches, grid2, grid_origin, grid_spacing
    )

    corrected_epipolar_error = corrected_matches[:, 1] - corrected_matches[:, 3]

    logging.info(
        "Epipolar error after correction: mean = {:.3f} pix., "
        "standard deviation = {:.3f} pix., max = {:.3f} pix.".format(
            np.mean(corrected_epipolar_error),
            np.std(corrected_epipolar_error),
            np.max(np.fabs(corrected_epipolar_error)),
        )
    )

    # TODO: add stats in content.json
    out_left_grid = os.path.join(out_dir, "left_epipolar_grid.tif")
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.LEFT_EPIPOLAR_GRID_TAG] = out_left_grid
    grids.write_grid(grid1, out_left_grid, grid_origin, grid_spacing)

    # Export corrected right grid
    out_right_grid = os.path.join(out_dir, "right_epipolar_grid.tif")
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.RIGHT_EPIPOLAR_GRID_TAG] = out_right_grid
    grids.write_grid(
        corrected_right_grid, out_right_grid, grid_origin, grid_spacing
    )

    # Export uncorrected right grid
    logging.info("Writing uncorrected right grid")
    out_right_grid_uncorrected = os.path.join(
        out_dir, "right_epipolar_grid_uncorrected.tif"
    )
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][
        output_prepare.RIGHT_EPIPOLAR_UNCORRECTED_GRID_TAG
    ] = out_right_grid_uncorrected
    grids.write_grid(
        grid2, out_right_grid_uncorrected, grid_origin, grid_spacing
    )

    # Compute the disparity range (we filter matches that are too off epipolar
    # lins after correction)
    corrected_std = np.std(corrected_epipolar_error)

    corrected_matches = corrected_matches[
        np.fabs(corrected_epipolar_error) < 3 * corrected_std
    ]
    logging.info(
        "{} matches discarded because "
        "their epipolar error is greater than 3*stdev of epipolar error "
        "after correction (3*stddev = {:.3f} pix.)".format(
            nb_matches - corrected_matches.shape[0], 3 * corrected_std
        )
    )

    logging.info(
        "Number of matches kept "
        "for disparity range estimation: {} matches".format(
            corrected_matches.shape[0]
        )
    )

    dmin, dmax = sparse_matching.compute_disparity_range(
        corrected_matches,
        static_conf.get_disparity_outliers_rejection_percent(),
    )
    margin = abs(dmax - dmin) * disparity_margin
    dmin -= margin
    dmax += margin
    logging.info(
        "Disparity range with margin: [{:.3f} pix., {:.3f} pix.] "
        "(margin = {:.3f} pix.)".format(dmin, dmax, margin)
    )
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.MINIMUM_DISPARITY_TAG] = dmin
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.MAXIMUM_DISPARITY_TAG] = dmax

    logging.info(
        "Equivalent range in meters: [{:.3f} m, {:.3f} m] "
        "(margin = {:.3f} m)".format(
            dmin * disp_to_alt_ratio,
            dmax * disp_to_alt_ratio,
            margin * disp_to_alt_ratio,
        )
    )

    # Export matches
    logging.info("Writing matches file")
    matches_array_path = os.path.join(out_dir, "matches.npy")
    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.MATCHES_TAG] = matches_array_path
    np.save(matches_array_path, corrected_matches)

    # Devibration part :
    # 1. Compute low resolution DSM from sparse matching matches
    # 2. Get Initial DEM (typically SRTM) on the same grid resolution
    # 3. Correction estimation of  DSM difference (with splines)
    # 4. Compute corrected low resolution DSM
    #    and corrected disparity to use in compute_dsm pipeline align mode.

    # First, triangulate matches
    points_cloud_from_matches = triangulation.triangulate_matches(
        out_json, corrected_matches
    )

    # Then define the size of the lower res DSM to rasterize
    low_res_dsm_params = static_conf.get_low_res_dsm_params()
    lowres_dsm_resolution = getattr(
        low_res_dsm_params,
        static_conf.low_res_dsm_resolution_in_degree_tag,  # Value in degree
    )
    lowres_dsm_sizex = int(
        math.ceil((inter_xmax - inter_xmin) / lowres_dsm_resolution)
    )
    lowres_dsm_sizey = int(
        math.ceil((inter_ymax - inter_ymin) / lowres_dsm_resolution)
    )
    logging.info(
        "Generating low resolution ({}°) DSM"
        " from matches ({}x{})".format(
            lowres_dsm_resolution, lowres_dsm_sizex, lowres_dsm_sizey
        )
    )

    lowres_dsm = rasterization.simple_rasterization_dataset(
        [points_cloud_from_matches],
        lowres_dsm_resolution,
        4326,
        color_list=None,
        xstart=inter_xmin,
        ystart=inter_ymax,
        xsize=lowres_dsm_sizex,
        ysize=lowres_dsm_sizey,
    )

    lowres_dsm_file = os.path.join(out_dir, "lowres_dsm_from_matches.nc")
    # TODO add proper CRS info
    lowres_dsm.to_netcdf(lowres_dsm_file)

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.LOWRES_DSM_TAG] = lowres_dsm_file

    # Now read the exact same grid on initial DEM
    lowres_initial_dem = otb_pipelines.read_lowres_dem(
        startx=inter_xmin,
        starty=inter_ymax,
        sizex=lowres_dsm_sizex,
        sizey=lowres_dsm_sizey,
        dem=srtm_dir,
        default_alt=default_alt,
        geoid=static_conf.get_geoid_path(),
        resolution=lowres_dsm_resolution,
    )
    lowres_initial_dem_file = os.path.join(out_dir, "lowres_initial_dem.nc")
    lowres_initial_dem.to_netcdf(lowres_initial_dem_file)

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][output_prepare.LOWRES_INITIAL_DEM_TAG] = lowres_initial_dem_file

    # also write the difference
    lowres_elevation_difference_file = os.path.join(
        out_dir, "lowres_elevation_diff.nc"
    )
    lowres_dsm_diff = lowres_initial_dem - lowres_dsm
    (lowres_dsm_diff).to_netcdf(lowres_elevation_difference_file)

    out_json[output_prepare.PREPROCESSING_SECTION_TAG][
        output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
    ][
        output_prepare.LOWRES_ELEVATION_DIFFERENCE_TAG
    ] = lowres_elevation_difference_file

    # Now, estimate a correction to align DSM on the lowres initial DEM
    splines = None
    cfg_low_res_dsm_min_sizex = getattr(
        low_res_dsm_params, static_conf.low_res_dsm_min_sizex_for_align_tag
    )

    cfg_low_res_dsm_min_sizey = getattr(
        low_res_dsm_params, static_conf.low_res_dsm_min_sizey_for_align_tag
    )

    if (
        lowres_dsm_sizex > cfg_low_res_dsm_min_sizex
        and lowres_dsm_sizey > cfg_low_res_dsm_min_sizey
    ):

        logging.info(
            "Estimating correction "
            "between low resolution DSM and initial DEM"
        )

        # First, we estimate direction of acquisition time for both images
        time_direction_vector, _, _ = projection.acquisition_direction(
            config, srtm_dir
        )

        origin = [
            float(lowres_dsm_diff[cst.X][0].values),
            float(lowres_dsm_diff[cst.Y][0].values),
        ]
        out_json[output_prepare.PREPROCESSING_SECTION_TAG][
            output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
        ][output_prepare.TIME_DIRECTION_LINE_ORIGIN_X_TAG] = origin[0]

        out_json[output_prepare.PREPROCESSING_SECTION_TAG][
            output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
        ][output_prepare.TIME_DIRECTION_LINE_ORIGIN_Y_TAG] = origin[1]

        out_json[output_prepare.PREPROCESSING_SECTION_TAG][
            output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
        ][
            output_prepare.TIME_DIRECTION_LINE_VECTOR_X_TAG
        ] = time_direction_vector[
            0
        ]
        out_json[output_prepare.PREPROCESSING_SECTION_TAG][
            output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
        ][
            output_prepare.TIME_DIRECTION_LINE_VECTOR_Y_TAG
        ] = time_direction_vector[
            1
        ]

        # Then we estimate the correction splines
        splines = devib.lowres_initial_dem_splines_fit(
            lowres_dsm,
            lowres_initial_dem,
            origin,
            time_direction_vector,
            ext=getattr(low_res_dsm_params, static_conf.low_res_dsm_ext_tag),
            order=getattr(
                low_res_dsm_params, static_conf.low_res_dsm_order_tag
            ),
        )

    else:
        logging.info(
            "Low resolution DSM is not large enough "
            "(minimum size is {}x{}) "
            "to estimate correction "
            "to fit initial DEM, skipping ...".format(
                cfg_low_res_dsm_min_sizex, cfg_low_res_dsm_min_sizey
            )
        )

    if splines is not None:
        # Save model to file
        lowres_dem_splines_fit_file = os.path.join(
            out_dir, "lowres_dem_splines_fit.pck"
        )

        with open(lowres_dem_splines_fit_file, "wb") as splines_fit_file_reader:
            pickle.dump(splines, splines_fit_file_reader)
            out_json[output_prepare.PREPROCESSING_SECTION_TAG][
                output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
            ][
                output_prepare.LOWRES_DEM_SPLINES_FIT_TAG
            ] = lowres_dem_splines_fit_file

            logging.info("Generating corrected low resolution DSM from matches")

            # Estimate correction on point cloud from matches
            points_cloud_from_matches_z_correction = splines(
                projection.project_coordinates_on_line(
                    points_cloud_from_matches.x,
                    points_cloud_from_matches.y,
                    origin,
                    time_direction_vector,
                )
            )

            # Estimate disparity correction
            points_cloud_disp_correction = (
                points_cloud_from_matches_z_correction / disp_to_alt_ratio
            )

            # Correct matches disparity
            z_corrected_matches = corrected_matches
            z_corrected_matches[:, 2] = (
                z_corrected_matches[:, 2] - points_cloud_disp_correction[:, 0]
            )

            # Triangulate and rasterize again
            corrected_points_cloud_from_matches = (
                triangulation.triangulate_matches(out_json, z_corrected_matches)
            )

            corrected_lowres_dsm = rasterization.simple_rasterization_dataset(
                [corrected_points_cloud_from_matches],
                lowres_dsm_resolution,
                corrected_points_cloud_from_matches.attrs["epsg"],
                xstart=inter_xmin,
                ystart=inter_ymax,
                xsize=lowres_dsm_sizex,
                ysize=lowres_dsm_sizey,
            )

            # Write corrected lowres dsm
            corrected_lowres_dsm_file = os.path.join(
                out_dir, "corrected_lowres_dsm_from_matches.nc"
            )

            # TODO add proper CRS info
            corrected_lowres_dsm.to_netcdf(corrected_lowres_dsm_file)
            out_json[output_prepare.PREPROCESSING_SECTION_TAG][
                output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
            ][
                output_prepare.CORRECTED_LOWRES_DSM_TAG
            ] = corrected_lowres_dsm_file

            # also write the difference
            corrected_lowres_elevation_difference_file = os.path.join(
                out_dir, "corrected_lowres_elevation_diff.nc"
            )
            corrected_lowres_dsm_diff = (
                lowres_initial_dem - corrected_lowres_dsm
            )
            (corrected_lowres_dsm_diff).to_netcdf(
                corrected_lowres_elevation_difference_file
            )
            out_json[output_prepare.PREPROCESSING_SECTION_TAG][
                output_prepare.PREPROCESSING_OUTPUT_SECTION_TAG
            ][
                output_prepare.CORRECTED_LOWRES_ELEVATION_DIFFERENCE_TAG
            ] = corrected_lowres_elevation_difference_file

    # Write the output json
    try:
        inputs.check_json(out_json, output_prepare.content_schema())
    except CheckerError as check_error:
        logging.warning(
            "content.json does not comply with schema: {}".format(check_error)
        )

    out_json_path = os.path.join(out_dir, "content.json")
    output_prepare.write_preprocessing_content_file(out_json, out_json_path)

    # stop cluster
    stop_cluster(mode, cluster, client)

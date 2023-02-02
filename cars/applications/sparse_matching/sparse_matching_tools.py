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
Sparse matching Sift module:
contains sift sparse matching method
"""

# Standard imports
from __future__ import absolute_import

import logging
import os

# Third party imports
import numpy as np
import pandas

# CARS imports
import cars.applications.sparse_matching.sparse_matching_constants as sm_cst
from cars.applications import application_constants
from cars.applications.point_cloud_outliers_removing import (
    outlier_removing_tools,
)
from cars.applications.triangulation import triangulation_tools
from cars.core import constants as cst
from cars.core import preprocessing, projection
from cars.externals import otb_pipelines


def dataset_matching(
    ds1,
    ds2,
    matching_threshold=0.6,
    n_octave=8,
    n_scale_per_octave=3,
    dog_threshold=20,
    edge_threshold=5,
    magnification=2.0,
    backmatching=True,
):
    """
    Compute sift matches between two datasets
    produced by stereo.epipolar_rectify_images

    :param ds1: Left image dataset
    :type ds1: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param ds2: Right image dataset
    :type ds2: xarray.Dataset as produced by stereo.epipolar_rectify_images
    :param threshold: Threshold for matches
    :type threshold: float
    :param backmatching: Also check that right vs. left gives same match
    :type backmatching: bool
    :return: matches
    :rtype: numpy buffer of shape (nb_matches,4)
    """
    size1 = [
        int(ds1.attrs["region"][2] - ds1.attrs["region"][0]),
        int(ds1.attrs["region"][3] - ds1.attrs["region"][1]),
    ]
    roi1 = [0, 0, size1[0], size1[1]]
    origin1 = [float(ds1.attrs["region"][0]), float(ds1.attrs["region"][1])]

    size2 = [
        int(ds2.attrs["region"][2] - ds2.attrs["region"][0]),
        int(ds2.attrs["region"][3] - ds2.attrs["region"][1]),
    ]
    roi2 = [0, 0, size2[0], size2[1]]
    origin2 = [float(ds2.attrs["region"][0]), float(ds2.attrs["region"][1])]

    matches = otb_pipelines.epipolar_sparse_matching(
        ds1,
        roi1,
        size1,
        origin1,
        ds2,
        roi2,
        size2,
        origin2,
        matching_threshold,
        n_octave,
        n_scale_per_octave,
        dog_threshold,
        edge_threshold,
        magnification,
        backmatching,
    )

    return matches


def remove_epipolar_outliers(matches, percent=0.1):
    # TODO used only in test functions to test compute_disparity_range
    # Refactor with sparse_matching
    """
    This function will filter the match vector
    according to a quantile of epipolar error
    used for testing compute_disparity_range sparse method

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema
    :type percent: float
    :return: the filtered match array
    :rtype: numpy array
    """
    epipolar_error_min = np.percentile(matches[:, 1] - matches[:, 3], percent)
    epipolar_error_max = np.percentile(
        matches[:, 1] - matches[:, 3], 100 - percent
    )
    logging.info(
        "Epipolar error range after outlier rejection: [{},{}]".format(
            epipolar_error_min, epipolar_error_max
        )
    )
    out = matches[(matches[:, 1] - matches[:, 3]) < epipolar_error_max]
    out = out[(out[:, 1] - out[:, 3]) > epipolar_error_min]

    return out


def compute_disparity_range(matches, percent=0.1):
    # TODO: Refactor with dense_matching to have only one API ?
    """
    This function will compute the disparity range
    from matches by filtering percent outliers

    :param matches: the [4,N] matches array
    :type matches: numpy array
    :param percent: the quantile to remove at each extrema (in %)
    :type percent: float
    :return: the disparity range
    :rtype: float, float
    """
    disparity = matches[:, 2] - matches[:, 0]

    mindisp = np.percentile(disparity, percent)
    maxdisp = np.percentile(disparity, 100 - percent)

    return mindisp, maxdisp


def compute_disp_min_disp_max(
    sensor_image_right,
    sensor_image_left,
    grid_left,
    corrected_grid_right,
    grid_right,
    matches,
    orchestrator,
    geometry_loader,
    srtm_dir,
    default_alt,
    pair_folder="",
    disp_margin=0.1,
    pair_key=None,
    disp_to_alt_ratio=None,
    save_matches=False,
):
    """
    Compute disp min and disp max from triangulated and filtered matches

    :param sensor_image_right: sensor image right
    :type sensor_image_right: CarsDataset
    :param sensor_image_left: sensor image left
    :type sensor_image_left: CarsDataset
    :param grid_left: grid left
    :type grid_left: CarsDataset CarsDataset
    :param corrected_grid_right: corrected grid right
    :type corrected_grid_right: CarsDataset
    :param grid_right: uncorrected grid right
    :type grid_right: CarsDataset
    :param matches: matches
    :type matches: np.ndarray
    :param orchestrator: orchestrator used
    :type orchestrator: Orchestrator
    :param geometry_loader: geometry loader to use
    :type geometry_loader: str
    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param default_alt: default altitude
    :type default_alt: float
    :param pair_folder: folder used for current pair
    :type pair_folder: str
    :param disp_margin: disparity margin
    :type disp_margin: float
    :param disp_to_alt_ratio: used for logging info
    :type disp_to_alt_ratio: float
    :param save_matches: true is matches needs to be saved
    :type save_matches: bool

    :return: disp min and disp max
    :rtype: float, float
    """
    input_stereo_cfg = (
        preprocessing.create_former_cars_post_prepare_configuration(
            sensor_image_left,
            sensor_image_right,
            grid_left,
            corrected_grid_right,
            pair_folder,
            uncorrected_grid_right=grid_right,
            srtm_dir=srtm_dir,
            default_alt=default_alt,
        )
    )

    point_cloud = triangulation_tools.triangulate_matches(
        geometry_loader, input_stereo_cfg, matches
    )

    # compute epsg
    epsg = preprocessing.compute_epsg(
        sensor_image_left,
        sensor_image_right,
        grid_left,
        corrected_grid_right,
        geometry_loader,
        orchestrator=orchestrator,
        pair_folder=pair_folder,
        srtm_dir=srtm_dir,
        default_alt=default_alt,
        disp_min=0,
        disp_max=0,
    )
    # Project point cloud to UTM
    projection.points_cloud_conversion_dataset(point_cloud, epsg)

    # Convert point cloud to pandas format to allow statistical filtering
    labels = [cst.X, cst.Y, cst.Z, cst.DISPARITY, cst.POINTS_CLOUD_CORR_MSK]
    cloud_array = []
    cloud_array.append(point_cloud[cst.X].values)
    cloud_array.append(point_cloud[cst.Y].values)
    cloud_array.append(point_cloud[cst.Z].values)
    cloud_array.append(point_cloud[cst.DISPARITY].values)
    cloud_array.append(point_cloud[cst.POINTS_CLOUD_CORR_MSK].values)
    pd_cloud = pandas.DataFrame(
        np.transpose(np.array(cloud_array)), columns=labels
    )

    # Statistical filtering
    filter_cloud, _ = outlier_removing_tools.statistical_outliers_filtering(
        pd_cloud, k=25, std_factor=3.0
    )

    # Export filtered matches
    matches_array_path = None
    if save_matches:
        logging.info("Writing matches file")
        filt_matches = np.array(filter_cloud.iloc[:, 0:4])
        if pair_folder is None:
            current_out_dir = orchestrator.out_dir
        else:
            current_out_dir = pair_folder
        matches_array_path = os.path.join(current_out_dir, "matches.npy")
        np.save(matches_array_path, filt_matches)

    # Obtain dmin dmax
    filt_disparity = np.array(filter_cloud.iloc[:, 3])
    dmax = np.max(filt_disparity)
    dmin = np.min(filt_disparity)

    margin = abs(dmax - dmin) * disp_margin
    dmin -= margin
    dmax += margin

    logging.info(
        "Disparity range with margin: [{:.3f} pix., {:.3f} pix.] "
        "(margin = {:.3f} pix.)".format(dmin, dmax, margin)
    )

    if disp_to_alt_ratio is not None:
        logging.info(
            "Equivalent range in meters: [{:.3f} m, {:.3f} m] "
            "(margin = {:.3f} m)".format(
                dmin * disp_to_alt_ratio,
                dmax * disp_to_alt_ratio,
                margin * disp_to_alt_ratio,
            )
        )

    # update orchestrator_out_json
    updating_infos = {
        application_constants.APPLICATION_TAG: {
            pair_key: {
                sm_cst.DISPARITY_RANGE_COMPUTATION_TAG: {
                    sm_cst.DISPARITY_MARGIN_PARAM_TAG: disp_margin,
                    sm_cst.MINIMUM_DISPARITY_TAG: dmin,
                    sm_cst.MAXIMUM_DISPARITY_TAG: dmax,
                    sm_cst.MATCHES_TAG: matches_array_path,
                }
            }
        }
    }
    orchestrator.update_out_info(updating_infos)

    return dmin, dmax

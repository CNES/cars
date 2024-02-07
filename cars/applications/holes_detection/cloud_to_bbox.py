# !/usr/bin/env python
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
CARS holes detection module init file
"""

import logging

# Standard imports
import os

# Third party imports
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.holes_detection import holes_detection_tools
from cars.applications.holes_detection.holes_detection import HolesDetection

# CARS imports
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, cars_dict


class CloudToBbox(
    HolesDetection, short_name="cloud_to_bbox"
):  # pylint: disable=R0903
    """
    CloudToBbox
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of CloudToBbox

        :param conf: configuration for holes detection
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # check conf

        # get rasterization parameter
        self.used_method = self.used_config["method"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf

        # get rasterization parameter
        overloaded_conf["method"] = conf.get("method", "cloud_to_bbox")

        holes_detection_schema = {
            "method": str,
        }

        # Check conf
        checker = Checker(holes_detection_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        epipolar_images_left,
        epipolar_images_right,
        classification=None,
        margin=0,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using plane method.

        :param epipolar_images_left:  left epipolar image
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right:  right epipolar image
        :type epipolar_images_right: CarsDataset
        :param is_activated:  activate application
        :type is_activated: bool
        :param margin: margin to use
        :type margin: int
        :param classification: mask classes to use
        :type classification: list(str)
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str

        :return: left holes, right holes
        :rtype: Tuple(CarsDataset, CarsDataset)

        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        if epipolar_images_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            left_bbox_cars_ds = cars_dataset.CarsDataset(
                "dict", name="cloud_to_bbox_left_" + pair_key
            )
            left_bbox_cars_ds.create_empty_copy(epipolar_images_left)
            left_bbox_cars_ds.overlaps *= 0

            right_bbox_cars_ds = cars_dataset.CarsDataset(
                "dict", name="cloud_to_bbox_left_" + pair_key
            )
            right_bbox_cars_ds.create_empty_copy(epipolar_images_right)
            right_bbox_cars_ds.overlaps *= 0

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {pair_key: {}}
            }
            self.orchestrator.update_out_info(updating_dict)
            logging.info(
                "Compute bbox: number tiles: {}".format(
                    epipolar_images_left.shape[1]
                    * epipolar_images_left.shape[0]
                )
            )

            if classification not in (None, []):
                # Get saving infos in order to save tiles when they are computed
                [
                    saving_info_left,
                    saving_info_right,
                ] = self.orchestrator.get_saving_infos(
                    [left_bbox_cars_ds, right_bbox_cars_ds]
                )

                # Add to replace list so tiles will be readble at the same time
                self.orchestrator.add_to_replace_lists(
                    left_bbox_cars_ds, cars_ds_name="epi_msk_bbox_left"
                )
                self.orchestrator.add_to_replace_lists(
                    right_bbox_cars_ds, cars_ds_name="epi_msk_bbox_right"
                )

                # Generate disparity maps
                for col in range(epipolar_images_left.shape[1]):
                    for row in range(epipolar_images_left.shape[0]):
                        if (epipolar_images_left[row, col] is not None) or (
                            epipolar_images_right[row, col] is not None
                        ):
                            # update saving_info with row and col needed for
                            # replacement
                            full_saving_info_left = ocht.update_saving_infos(
                                saving_info_left, row=row, col=col
                            )
                            full_saving_info_right = ocht.update_saving_infos(
                                saving_info_right, row=row, col=col
                            )

                            # get window and overlaps
                            window_left = epipolar_images_left.tiling_grid[
                                row, col, :
                            ]
                            window_right = epipolar_images_right.tiling_grid[
                                row, col, :
                            ]
                            overlap_left = epipolar_images_left.overlaps[
                                row, col, :
                            ]
                            overlap_right = epipolar_images_right.overlaps[
                                row, col, :
                            ]

                            # Compute bbox
                            (
                                left_bbox_cars_ds[row, col],
                                right_bbox_cars_ds[row, col],
                            ) = self.orchestrator.cluster.create_task(
                                compute_mask_bboxes_wrapper, nout=2
                            )(
                                epipolar_images_left[row, col],
                                epipolar_images_right[row, col],
                                window_left,
                                window_right,
                                overlap_left,
                                overlap_right,
                                classification,
                                saving_info_left=full_saving_info_left,
                                saving_info_right=full_saving_info_right,
                            )
        else:
            logging.error(
                "CloudToBbox application doesn't "
                "support this input data format"
            )

        return left_bbox_cars_ds, right_bbox_cars_ds


def compute_mask_bboxes_wrapper(
    left_image_dataset,
    right_image_dataset,
    window_left,
    window_right,
    overlap_left,
    overlap_right,
    classification,
    margin=20,
    saving_info_left=None,
    saving_info_right=None,
):
    """
    Compute mask bounding boxes.

    :param left_image_dataset: tiled Left image
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :type left_image_dataset: xr.Dataset
      - dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :param right_image_dataset: tiled Right image
    :type right_image_dataset: xr.Dataset
    :param window_left: left window
    :type window_left: dict
    :param window_right: right window
    :type window_right: dict
    :param overlap_left: left  overlpas
    :type overlap_left: dict
    :param overlap_right: right overlaps
    :type overlap_right: dict
    :param classification: mask classes to use
    :type classification: list(str)
    :param margin: margin to use
    :type margin: int
    :param saving_info_left: saving infos left
    :type saving_info_left: dict
    :param saving_info_right: saving infos right
    :type saving_info_right: dict

    :return: Left image object, Right image object (if exists)

    Returned objects are composed of dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    """

    # compute offsets
    row_offset_left = window_left[0] - overlap_left[0]
    col_offset_left = window_left[2] - overlap_left[2]
    row_offset_right = window_right[0] - overlap_right[0]
    col_offset_right = window_right[2] - overlap_right[2]

    bbox_left = {}

    if left_image_dataset is not None:
        bbox_left = holes_detection_tools.localize_masked_areas(
            left_image_dataset,
            classification,
            row_offset=row_offset_left,
            col_offset=col_offset_left,
            margin=margin,
        )

    bbox_right = {}

    if right_image_dataset is not None:
        bbox_right = holes_detection_tools.localize_masked_areas(
            right_image_dataset,
            classification,
            row_offset=row_offset_right,
            col_offset=col_offset_right,
            margin=margin,
        )

    # add saving infos
    bbox_left_dict = cars_dict.CarsDict({"list_bbox": bbox_left})
    cars_dataset.fill_dict(bbox_left_dict, saving_info=saving_info_left)

    bbox_right_dict = cars_dict.CarsDict({"list_bbox": bbox_right})
    cars_dataset.fill_dict(bbox_right_dict, saving_info=saving_info_right)

    return bbox_left_dict, bbox_right_dict

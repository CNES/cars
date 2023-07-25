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
this module contains the epipolar cloud fusion application class.
"""


# Standard imports
import logging
import os
from collections import Counter

# Third party imports
import numpy as np
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_fusion import (
    cloud_fusion_constants,
    pc_tif_tools,
    point_cloud_tools,
)
from cars.applications.point_cloud_fusion.point_cloud_fusion import (
    PointCloudFusion,
)
from cars.core import projection, tiling
from cars.data_structures import cars_dataset


class MappingToTerrainTiles(
    PointCloudFusion, short_name="mapping_to_terrain_tiles"
):
    """
    EpipolarCloudFusion
    """

    def __init__(self, conf=None):
        """
        Init function of EpipolarCloudFusion

        :param conf: configuration for fusion
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

        # Cloud fusion
        self.used_method = self.used_config["method"]

        # check loader

        # Saving files
        self.save_points_cloud_as_laz = self.used_config.get(
            "save_points_cloud_as_laz", False
        )
        self.save_points_cloud_as_csv = self.used_config.get(
            "save_points_cloud_as_csv", False
        )

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
        overloaded_conf["method"] = conf.get(
            "method", "mapping_to_terrain_tiles"
        )

        overloaded_conf["save_points_cloud_as_laz"] = conf.get(
            "save_points_cloud_as_laz", False
        )
        overloaded_conf["save_points_cloud_as_csv"] = conf.get(
            "save_points_cloud_as_csv", False
        )

        points_cloud_fusion_schema = {
            "method": str,
            "save_points_cloud_as_laz": bool,
            "save_points_cloud_as_csv": bool,
        }

        # Check conf
        checker = Checker(points_cloud_fusion_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        list_epipolar_points_cloud,
        bounds,
        epsg,
        orchestrator=None,
        margins=0,
        optimal_terrain_tile_width=500,
    ):
        """
        Run EpipolarCloudFusion application.

        Creates a CarsDataset corresponding to the merged points clouds,
        tiled with the terrain grid used during rasterization.

        :param list_epipolar_points_cloud: list with points clouds\
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "color", "msk",
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound" \
                "elevation_delta_lower_bound", "elevation_delta_upper_bound"
        :type list_epipolar_points_cloud: list(CarsDataset) filled with
          xr.Dataset
        :param bounds: terrain bounds
        :type bounds: list
        :param epsg: epsg to use
        :type epsg: str
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param margins: margins needed for tiles, meter or degree
        :type margins: float
        :param optimal_terrain_tile_width: optimal terrain tile width
        :type optimal_terrain_tile_width: int

        :return: Merged points clouds

            CarsDataset contains:

            - Z x W Delayed tiles\
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j","idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing: "bounds", "epsg"

        :rtype: CarsDataset filled with pandas.DataFrame

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

        # Compute bounds and terrain grid
        [xmin, ymin, xmax, ymax] = bounds

        # Split terrain bounding box in pieces
        terrain_tiling_grid = tiling.generate_tiling_grid(
            xmin,
            ymin,
            xmax,
            ymax,
            optimal_terrain_tile_width,
            optimal_terrain_tile_width,
        )

        # Get dataset type of first item in list_epipolar_points_cloud
        pc_dataset_type = list_epipolar_points_cloud[0].dataset_type

        if pc_dataset_type in (
            "arrays",
            "dict",
            "points",
        ):
            # Create CarsDataset
            merged_point_cloud = cars_dataset.CarsDataset("points")

            # Compute tiling grid
            merged_point_cloud.tiling_grid = terrain_tiling_grid

            # update attributes
            merged_point_cloud.attributes["bounds"] = bounds
            merged_point_cloud.attributes["epsg"] = epsg

            number_of_terrain_tiles = (
                merged_point_cloud.tiling_grid.shape[1]
                * merged_point_cloud.tiling_grid.shape[0]
            )

            logging.info(
                "Number of tiles in cloud fusion :"
                "row : {} "
                "col : {}".format(
                    merged_point_cloud.shape[0],
                    merged_point_cloud.shape[1],
                )
            )

            number_of_epipolar_tiles_per_terrain_tiles = []

            if pc_dataset_type in (
                "arrays",
                "points",
            ):
                # deal with delayed tiles, with a priori disp min and max

                # Add epipolar_points_min and epipolar_points_max used
                #  in point_cloud_fusion
                # , to get corresponding tiles (terrain)
                # TODO change method for corresponding tiles
                list_points_min = []
                list_points_max = []
                for points_cloud in list_epipolar_points_cloud:
                    points_min, points_max = tiling.terrain_grid_to_epipolar(
                        terrain_tiling_grid,
                        points_cloud.tiling_grid,
                        points_cloud.attributes["epipolar_grid_min"],
                        points_cloud.attributes["epipolar_grid_max"],
                        epsg,
                    )
                    list_points_min.append(points_min)
                    list_points_max.append(points_max)

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    cloud_fusion_constants.CLOUD_FUSION_RUN_TAG: {
                        cloud_fusion_constants.EPSG_TAG: epsg,
                        cloud_fusion_constants.MARGINS_TAG: margins,
                        cloud_fusion_constants.NUMBER_TERRAIN_TILES: (
                            number_of_terrain_tiles
                        ),
                        cloud_fusion_constants.BOUNDS: bounds,
                    },
                }
            }
            orchestrator.update_out_info(updating_dict)

            # Generate merged point clouds
            logging.info(
                "Point clouds: Merged points number: {}".format(
                    merged_point_cloud.shape[1] * merged_point_cloud.shape[0]
                )
            )

            # Compute corresponing tiles in parallel if from tif files
            if pc_dataset_type == "dict":
                corresponding_tiles_cars_ds = (
                    pc_tif_tools.get_corresponding_tiles_tif(
                        terrain_tiling_grid,
                        list_epipolar_points_cloud,
                        margins=margins,
                        orchestrator=self.orchestrator,
                    )
                )

            # Save objects

            if self.save_points_cloud_as_csv or self.save_points_cloud_as_laz:
                # Points cloud file name
                # TODO in input conf file
                pc_file_name = os.path.join(
                    self.orchestrator.out_dir, "points_cloud"
                )
                self.orchestrator.add_to_save_lists(
                    pc_file_name,
                    None,
                    merged_point_cloud,
                    cars_ds_name="merged_points_cloud",
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [merged_point_cloud]
            )

            for col in range(merged_point_cloud.shape[1]):
                for row in range(merged_point_cloud.shape[0]):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    if pc_dataset_type in (
                        "arrays",
                        "points",
                    ):
                        # Get required point clouds
                        (
                            terrain_region,
                            required_point_clouds,
                            _rank,
                            _pos,
                        ) = tiling.get_corresponding_tiles_row_col(
                            terrain_tiling_grid,
                            row,
                            col,
                            list_epipolar_points_cloud,
                            list_points_min,
                            list_points_max,
                        )
                    else:
                        # Get correspondances previously computed
                        terrain_region = corresponding_tiles_cars_ds[row, col][
                            "terrain_region"
                        ]
                        required_point_clouds = corresponding_tiles_cars_ds[
                            row, col
                        ]["required_point_clouds"]

                    if (
                        len(
                            [
                                value
                                for value, _ in required_point_clouds
                                if not isinstance(value, type(None))
                            ]
                        )
                        > 0
                    ):
                        logging.debug(
                            "Number of clouds to process for this terrain"
                            " tile: {}".format(len(required_point_clouds))
                        )
                        number_of_epipolar_tiles_per_terrain_tiles.append(
                            len(required_point_clouds)
                        )

                        # Delayed call to rasterization operations using all
                        #  required point clouds
                        merged_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            compute_point_cloud_wrapper
                        )(
                            required_point_clouds,
                            epsg,
                            xmin=terrain_region[0],
                            ymin=terrain_region[1],
                            xmax=terrain_region[2],
                            ymax=terrain_region[3],
                            margins=margins,
                            save_pc_as_laz=self.save_points_cloud_as_laz,
                            save_pc_as_csv=self.save_points_cloud_as_csv,
                            saving_info=full_saving_info,
                        )

            # Sort tiles according to rank TODO remove or implement it ?

            # Add delayed_dsm_tiles to orchestrator
            logging.info(
                "Submitting {} tasks to dask".format(number_of_terrain_tiles)
            )

            logging.info(
                "Number of epipolar tiles "
                "for each terrain tile (counter): {}".format(
                    sorted(
                        Counter(
                            number_of_epipolar_tiles_per_terrain_tiles
                        ).items()
                    )
                )
            )

            logging.info(
                "Average number of epipolar tiles "
                "for each terrain tile: {}".format(
                    int(
                        np.round(
                            np.mean(number_of_epipolar_tiles_per_terrain_tiles)
                        )
                    )
                )
            )

            logging.info(
                "Max number of epipolar tiles "
                "for each terrain tile: {}".format(
                    np.max(number_of_epipolar_tiles_per_terrain_tiles)
                )
            )

        else:
            logging.error(
                "PointsCloudRasterisation application doesn't "
                "support this input data format"
            )

        return merged_point_cloud


def compute_point_cloud_wrapper(
    point_clouds,
    epsg,
    xmin: float = None,
    ymin: float = None,
    xmax: float = None,
    ymax: float = None,
    margins: float = 0,
    save_pc_as_laz: bool = False,
    save_pc_as_csv: bool = False,
    saving_info=None,
):
    """
    Wrapper for points clouds fusion step :
    - Convert a list of clouds to correct epsg

    :param point_clouds: list of clouds, list of (dataset, dataset_id) with :
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
    :type point_clouds: list((xr.Dataset, int))
    :param  epsg_code: epsg code for the CRS of the output DSM
    :type epsg_code: int
    :param  stereo_out_epsg: epsg code to convert point cloud to, if needed
    :type stereo_out_epsg: int
    :param xmin: xmin of the rasterization grid
        (if None, will be estimated by the function)
    :param xmin: xmin of the rasterization grid
        (if None, will be estimated by the function)
    :param xmax: xmax of the rasterization grid
        (if None, will be estimated by the function)
    :param ymax: ymax of the rasterization grid
        (if None, will be estimated by the function)
    :param margins: margins needed for tiles, meter or degree
    :type margins: float
    :param save_pc_as_laz: save point cloud as laz
    :type save_pc_as_laz: bool
    :param save_pc_as_csv: save point cloud as csv
    :type save_pc_as_csv: bool
    :param saving_info: informations about CarsDataset ID.
    :type saving_info: dict

    :return: merged points cloud dataframe with:
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
            - attrs : xmin, xmax, ymin, ymax, saving_info
    :rtype: pandas.DataFrame
    """
    # Remove None tiles
    clouds = []
    clouds_ids = []
    for value, pc_id in point_clouds:
        if value is not None:
            clouds.append(value)
            clouds_ids.append(pc_id)
    if len(clouds) == 0:
        raise RuntimeError("All clouds are None")

    # combine clouds
    if not isinstance(clouds[0], dict):
        pc_pandas, cloud_epsg = point_cloud_tools.create_combined_cloud(
            clouds,
            clouds_ids,
            epsg,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            margin=margins,
            epipolar_border_margin=0,
            with_coords=True,
        )
        # get color type list
        color_type = point_cloud_tools.get_color_type(clouds)
    else:
        # combined pc from tif files
        (
            pc_pandas,
            cloud_epsg,
            color_type,
        ) = pc_tif_tools.create_combined_cloud_from_tif(
            clouds,
            clouds_ids,
            epsg,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            margin=margins,
        )

    # Conversion to UTM
    if cloud_epsg != epsg:
        projection.points_cloud_conversion_dataframe(
            pc_pandas, cloud_epsg, epsg
        )
        cloud_epsg = epsg

    # Fill attributes for rasterization
    attributes = {
        "epsg": cloud_epsg,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "color_type": color_type,
        "save_points_cloud_as_laz": save_pc_as_laz,
        "save_points_cloud_as_csv": save_pc_as_csv,
    }
    cars_dataset.fill_dataframe(
        pc_pandas, saving_info=saving_info, attributes=attributes
    )

    return pc_pandas

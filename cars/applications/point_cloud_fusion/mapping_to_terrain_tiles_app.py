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
from shapely.geometry import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_fusion import (
    cloud_fusion_constants,
    pc_fusion_algo,
    pc_fusion_wrappers,
)
from cars.applications.point_cloud_fusion.abstract_pc_fusion_app import (
    PointCloudFusion,
)
from cars.applications.triangulation.triangulation_wrappers import (
    generate_point_cloud_file_names,
)
from cars.core import constants as cst
from cars.core import inputs, projection, tiling
from cars.core.utils import safe_makedirs
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

        overloaded_conf["save_by_pair"] = conf.get("save_by_pair", False)

        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )
        point_cloud_fusion_schema = {
            "method": str,
            "save_by_pair": bool,
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
        }

        # Check conf
        checker = Checker(point_cloud_fusion_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa: C901
        self,
        list_epipolar_point_clouds,
        bounds,
        epsg,
        source_pc_names=None,
        orchestrator=None,
        margins=0,
        optimal_terrain_tile_width=500,
        roi=None,
        save_laz_output=False,
    ):
        """
        Run EpipolarCloudFusion application.

        Creates a CarsDataset corresponding to the merged point clouds,
        tiled with the terrain grid used during rasterization.

        :param list_epipolar_point_clouds: list with point clouds\
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "texture", "msk",
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound" \
                "elevation_delta_lower_bound", "elevation_delta_upper_bound"
        :type list_epipolar_point_clouds: list(CarsDataset) filled with
          xr.Dataset
        :param bounds: terrain bounds
        :type bounds: list
        :param epsg: epsg to use
        :type epsg: str
        :param source_pc_names: source pc names
        :type source_pc_names: list[str]
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param margins: margins needed for tiles, meter or degree
        :type margins: float
        :param optimal_terrain_tile_width: optimal terrain tile width
        :type optimal_terrain_tile_width: int
        :param save_laz_output: save output point cloud as laz
        :type save_laz_output: bool


        :return: Merged point clouds

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

        save_point_cloud_as_csv = self.used_config.get(
            application_constants.SAVE_INTERMEDIATE_DATA, False
        )
        save_point_cloud_as_laz = (
            self.used_config.get(
                application_constants.SAVE_INTERMEDIATE_DATA, False
            )
            or save_laz_output
        )
        save_by_pair = self.used_config.get("save_by_pair", False)

        if source_pc_names is None:
            source_pc_names = ["PAIR_0"]

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
        source_pc_names = []
        for point_cloud in list_epipolar_point_clouds:
            if "source_pc_name" in point_cloud.attributes:
                source_pc_names.append(point_cloud.attributes["source_pc_name"])
        # Get dataset type of first item in list_epipolar_point_clouds
        pc_dataset_type = list_epipolar_point_clouds[0].dataset_type

        if pc_dataset_type in (
            "arrays",
            "dict",
            "points",
        ):
            # Create CarsDataset
            merged_point_cloud = cars_dataset.CarsDataset(
                "points", name="point_cloud_fusion"
            )

            # Compute tiling grid
            merged_point_cloud.tiling_grid = terrain_tiling_grid

            # update attributes
            merged_point_cloud.attributes["bounds"] = bounds
            merged_point_cloud.attributes["epsg"] = epsg
            merged_point_cloud.attributes["source_pc_names"] = source_pc_names

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
                for point_cloud in list_epipolar_point_clouds:
                    points_min, points_max = tiling.terrain_grid_to_epipolar(
                        terrain_tiling_grid,
                        point_cloud.tiling_grid,
                        point_cloud.attributes["epipolar_grid_min"],
                        point_cloud.attributes["epipolar_grid_max"],
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
            color_type = None
            if pc_dataset_type == "dict":
                corresponding_tiles_cars_ds = (
                    pc_fusion_algo.get_corresponding_tiles_tif(
                        terrain_tiling_grid,
                        list_epipolar_point_clouds,
                        margins=margins,
                        orchestrator=self.orchestrator,
                    )
                )
                color_file = list_epipolar_point_clouds[0].tiles[0][0]["data"][
                    "texture"
                ]
                if color_file is not None:
                    color_type = inputs.rasterio_get_image_type(color_file)
                    merged_point_cloud.attributes["color_type"] = color_type

            # Save objects
            csv_pc_dir_name = None
            if save_point_cloud_as_csv:
                # Point cloud file name
                csv_pc_dir_name = os.path.join(
                    self.orchestrator.out_dir,
                    "dump_dir",
                    "point_cloud_fusion",
                    "csv",
                )
                safe_makedirs(csv_pc_dir_name)
                self.orchestrator.add_to_compute_lists(
                    merged_point_cloud, cars_ds_name="merged_point_cloud_csv"
                )

            laz_pc_dir_name = None
            if save_point_cloud_as_laz:
                # Point cloud file name
                if save_laz_output:
                    laz_pc_dir_name = os.path.join(
                        self.orchestrator.out_dir, "point_cloud"
                    )
                else:
                    laz_pc_dir_name = os.path.join(
                        self.orchestrator.out_dir,
                        "dump_dir",
                        "point_cloud_fusion",
                        "laz",
                    )
                safe_makedirs(laz_pc_dir_name)
                self.orchestrator.add_to_compute_lists(
                    merged_point_cloud, cars_ds_name="merged_point_cloud"
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [merged_point_cloud]
            )
            pc_index = {}
            for col in range(merged_point_cloud.shape[1]):
                for row in range(merged_point_cloud.shape[0]):
                    # update saving infos for potential replacement
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
                            list_epipolar_point_clouds,
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

                    terrain_region_poly = Polygon(
                        [
                            [terrain_region[0], terrain_region[1]],
                            [terrain_region[0], terrain_region[3]],
                            [terrain_region[2], terrain_region[3]],
                            [terrain_region[2], terrain_region[1]],
                            [terrain_region[0], terrain_region[1]],
                        ]
                    )
                    if len(
                        [
                            value
                            for value, _ in required_point_clouds
                            if not isinstance(value, type(None))
                        ]
                    ) > 0 and (
                        roi is None or terrain_region_poly.intersects(roi)
                    ):
                        logging.debug(
                            "Number of clouds to process for this terrain"
                            " tile: {}".format(len(required_point_clouds))
                        )
                        number_of_epipolar_tiles_per_terrain_tiles.append(
                            len(required_point_clouds)
                        )

                        csv_pc_file_name, laz_pc_file_name = (
                            generate_point_cloud_file_names(
                                csv_pc_dir_name,
                                laz_pc_dir_name,
                                row,
                                col,
                                pc_index,
                                pair_key=(
                                    source_pc_names if save_by_pair else None
                                ),
                            )
                        )

                        # Delayed call to rasterization operations using all
                        # required point clouds
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
                            save_by_pair=save_by_pair,
                            point_cloud_csv_file_name=csv_pc_file_name,
                            point_cloud_laz_file_name=laz_pc_file_name,
                            saving_info=full_saving_info,
                            source_pc_names=source_pc_names,
                        )

            # update point cloud index
            if save_laz_output:
                self.orchestrator.update_index(pc_index)

            # Sort tiles according to rank TODO remove or implement it ?

            # Raise an error if no tiles has been found
            if len(number_of_epipolar_tiles_per_terrain_tiles) < 1:
                raise RuntimeError(
                    "No epipolar tiles has been found inside the ROI! "
                    "Please try with an other ROI."
                )

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
                "PointCloudRasterisation application doesn't "
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
    save_by_pair: bool = False,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
    saving_info=None,
    source_pc_names=None,
):
    """
    Wrapper for point clouds fusion step :
    - Convert a list of clouds to correct epsg

    :param point_clouds: list of clouds, list of (dataset, dataset_id) with :
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_TEXTURE
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
    :param save_by_pair: save point cloud as pair
    :type save_by_pair: bool
    :param point_cloud_csv_file_name: write point cloud as CSV in filename
        (if None, the point cloud is not written as csv)
    :type point_cloud_csv_file_name: str
    :param point_cloud_laz_file_name: write point cloud as laz in filename
        (if None, the point cloud is not written as laz)
    :type point_cloud_laz_file_name: str
    :param saving_info: informations about CarsDataset ID.
    :type saving_info: dict
    :param source_pc_names: source point cloud name (correspond to pair_key)
    :type source_pc_names: list str

    :return: merged point cloud dataframe with:
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_TEXTURE
            - attrs : xmin, xmax, ymin, ymax, saving_info
    :rtype: pandas.DataFrame
    """
    # Remove None tiles
    clouds = []
    clouds_ids = []
    disparity_range_is_cropped = False
    for value, pc_id in point_clouds:
        if value is not None:
            clouds.append(value)
            clouds_ids.append(pc_id)
            # Check if disparity range was cropped during process
            if ocht.get_disparity_range_cropped(value):
                disparity_range_is_cropped = True
    if len(clouds) == 0:
        raise RuntimeError("All clouds are None")

    # combine clouds
    if not isinstance(clouds[0], dict):
        pc_pandas, cloud_epsg = pc_fusion_algo.create_combined_cloud(
            clouds,
            clouds_ids,
            epsg,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            margin=margins,
            with_coords=True,
        )
        # get color type list
        color_type = pc_fusion_wrappers.get_color_type(clouds)
    else:
        # combined pc from tif files
        (
            pc_pandas,
            cloud_epsg,
            color_type,
        ) = pc_fusion_algo.create_combined_cloud_from_tif(
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
        projection.point_cloud_conversion_dataframe(pc_pandas, cloud_epsg, epsg)
        cloud_epsg = epsg

    # Fill attributes for rasterization
    attributes = {
        "epsg": cloud_epsg,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "color_type": color_type,
        "source_pc_names": source_pc_names,
        "number_of_pc": len(source_pc_names),
        cst.CROPPED_DISPARITY_RANGE: disparity_range_is_cropped,
    }
    cars_dataset.fill_dataframe(
        pc_pandas, saving_info=saving_info, attributes=attributes
    )

    # save point cloud in worker
    if point_cloud_csv_file_name:
        cars_dataset.run_save_points(
            pc_pandas,
            point_cloud_csv_file_name,
            save_by_pair=save_by_pair,
            overwrite=True,
            point_cloud_format="csv",
            overwrite_file_name=False,
        )
    if point_cloud_laz_file_name:
        cars_dataset.run_save_points(
            pc_pandas,
            point_cloud_laz_file_name,
            save_by_pair=save_by_pair,
            overwrite=True,
            point_cloud_format="laz",
            overwrite_file_name=False,
        )

    return pc_pandas

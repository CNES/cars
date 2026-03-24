"""
This module contains the implementation of a tiled Dense matching application
"""

import logging
import os

import numpy as np
import xarray as xr
from json_checker import And, Checker, Or
from rasterio.profiles import DefaultGTiffProfile

import cars.applications.dense_matching.dense_matching_constants as dm_cst
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matching.abstract_dense_matching_app import (
    AbstractDenseMatchingApplication,
)
from cars.applications.dense_matching.disparity_grid_algo import (
    generate_disp_range_const_tile_wrapper,
    generate_disp_range_from_dem_wrapper,
)

# CARS imports
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, format_transformation
from cars.data_structures.cars_dict import CarsDict
from cars.orchestrator.cluster.log_wrapper import cars_profile

# disable too many lines, too many instance attributes,
# too many positional arguments
# pylint:disable=C0302,R0902,R0917


class BasicDenseMatchingApplication(
    AbstractDenseMatchingApplication,
    short_name=["basic"],
):
    """
    Implementation of a tiled Dense matching application, with support for
    dense matching methods
    """

    def __init__(self, conf=None):

        self.schema = {
            "application": And(str, lambda x: x in self.available_applications),
            "min_epi_tile_size": And(int, lambda x: x > 0),
            "max_epi_tile_size": And(int, lambda x: x > 0),
            "epipolar_tile_margin_in_percent": int,
            "min_elevation_offset": Or(None, int),
            "max_elevation_offset": Or(None, int),
            "disp_min_threshold": Or(None, int),
            "disp_max_threshold": Or(None, int),
            "save_intermediate_data": bool,
            "use_global_disp_range": bool,
            "local_disp_grid_step": int,
            "disp_range_propagation_filter_size": And(
                Or(int, float), lambda x: x >= 0
            ),
            "epi_disp_grid_tile_size": int,
            "required_bands": [str],
        }

        super().__init__(conf=conf)

        self.used_config["application"] = "basic"

        # required for pylint
        self.orchestrator = None

        self.min_epi_tile_size = self.used_config["min_epi_tile_size"]
        self.max_epi_tile_size = self.used_config["max_epi_tile_size"]
        self.epipolar_tile_margin_in_percent = self.used_config[
            "epipolar_tile_margin_in_percent"
        ]
        self.min_elevation_offset = self.used_config["min_elevation_offset"]
        self.max_elevation_offset = self.used_config["max_elevation_offset"]
        self.disp_min_threshold = self.used_config["disp_min_threshold"]
        self.disp_max_threshold = self.used_config["disp_max_threshold"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]
        self.use_global_disp_range = self.used_config["use_global_disp_range"]
        self.local_disp_grid_step = self.used_config["local_disp_grid_step"]
        self.disp_range_propagation_filter_size = self.used_config[
            "disp_range_propagation_filter_size"
        ]
        self.epi_disp_grid_tile_size = self.used_config[
            "epi_disp_grid_tile_size"
        ]
        self.required_bands = self.used_config["required_bands"]

    @property
    def loader(self):
        return self.dense_matching_method.loader

    def check_conf(self, conf):
        """
        Merge user configuration with default values and validate schema.
        Extra keys in conf are preserved and ignored during schema validation.
        """
        # init conf
        if conf is None:
            conf = {}

        save_intermediate_data = conf.get("save_intermediate_data", False)

        # default configuration
        default_conf = {
            "application": "basic",
            "save_intermediate_data": save_intermediate_data,
            "min_epi_tile_size": 300,
            "max_epi_tile_size": 1500,
            "epipolar_tile_margin_in_percent": 60,
            "min_elevation_offset": None,
            "max_elevation_offset": None,
            "disp_min_threshold": None,
            "disp_max_threshold": None,
            "use_global_disp_range": False,
            "local_disp_grid_step": 10,
            "disp_range_propagation_filter_size": 50,
            "epi_disp_grid_tile_size": 800,
            "required_bands": ["b0"],
        }

        # merge defaults + user conf
        used_conf = default_conf.copy()
        used_conf.update(conf)
        used_conf.update(self.dense_matching_method.used_config)

        # merge expected schema with the used method's schema
        complete_schema = self.schema.copy()
        complete_schema.update(self.dense_matching_method.schema)

        checker = Checker(complete_schema)
        checker.validate(used_conf)

        # additional checks: min/max consistency
        min_epi_tile_size = used_conf["min_epi_tile_size"]
        max_epi_tile_size = used_conf["max_epi_tile_size"]
        if min_epi_tile_size > max_epi_tile_size:
            raise ValueError(
                "Maximal tile size should be bigger than "
                "minimal tile size for optimal tile size search"
            )

        min_elevation_offset = used_conf["min_elevation_offset"]
        max_elevation_offset = used_conf["max_elevation_offset"]
        if (
            min_elevation_offset is not None
            and max_elevation_offset is not None
            and min_elevation_offset > max_elevation_offset
        ):
            raise ValueError(
                "Maximal elevation should be bigger than "
                "minimal elevation for dense matching"
            )

        disp_min_threshold = used_conf["disp_min_threshold"]
        disp_max_threshold = used_conf["disp_max_threshold"]
        if (
            disp_min_threshold is not None
            and disp_max_threshold is not None
            and disp_min_threshold > disp_max_threshold
        ):
            raise ValueError(
                "Maximal disparity should be bigger than "
                "minimal disparity for dense matching"
            )

        return used_conf

    def get_optimal_tile_size(self, disp_range_grid, max_ram_per_worker):
        optimal_tile_size, local_tile_optimal_size_fun = (
            self.dense_matching_method.get_optimal_tile_size(
                disp_range_grid,
                max_ram_per_worker,
                self.min_epi_tile_size,
                self.max_epi_tile_size,
                self.local_disp_grid_step,
                self.epipolar_tile_margin_in_percent,
            )
        )
        return optimal_tile_size, local_tile_optimal_size_fun

    def get_required_bands(self):
        """
        Get bands required by this application

        :return: required bands for left and right image
        :rtype: dict
        """
        required_bands = {}
        required_bands["left"] = self.required_bands
        required_bands["right"] = self.required_bands
        return required_bands

    def get_performance_map_parameters(self):
        return self.dense_matching_method.get_performance_map_parameters()

    def get_margins_fun(self, grid_left, disp_range_grid):
        return self.dense_matching_method.get_margins_fun(
            grid_left,
            disp_range_grid,
            self.min_elevation_offset,
            self.max_elevation_offset,
        )

    def get_method(self):
        return self.dense_matching_method.get_method()

    @cars_profile(name="Disp Grid Generation")
    def generate_disparity_grids(  # noqa: C901
        self,
        sensor_image_right,
        grid_right,
        geom_plugin_with_dem_and_geoid,
        dmin=None,
        dmax=None,
        dem_min=None,
        dem_max=None,
        pair_folder=None,
        orchestrator=None,
    ):
        """
        Generate disparity grids min and max, with given step

        global mode: uses dmin and dmax
        local mode: uses dems


        :param sensor_image_right: sensor image right
        :type sensor_image_right: dict
        :param grid_right: right epipolar grid
        :type grid_right: dict
        :param geom_plugin_with_dem_and_geoid: geometry plugin with dem mean
            used to generate epipolar grids
        :type geom_plugin_with_dem_and_geoid: GeometryPlugin
        :param dmin: minimum disparity
        :type dmin: float
        :param dmax: maximum disparity
        :type dmax: float
        :param dem_min: path to minimum dem
        :type dem_min: str
        :param dem_max: path to maximum dem
        :type dem_max: str
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param orchestrator: orchestrator
        :type orchestrator: Orchestrator


        :return disparity grid range, containing grid min and max
        :rtype: CarsDataset
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be aware, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        epi_size_row = grid_right["epipolar_size_y"]
        epi_size_col = grid_right["epipolar_size_x"]
        disp_to_alt_ratio = grid_right["disp_to_alt_ratio"]

        # Generate grid array
        nb_rows = int(epi_size_row / self.local_disp_grid_step) + 1
        nb_cols = int(epi_size_col / self.local_disp_grid_step) + 1
        row_range, step_row = np.linspace(
            0, epi_size_row, nb_rows, retstep=True
        )
        col_range, step_col = np.linspace(
            0, epi_size_col, nb_cols, retstep=True
        )

        # Create CarsDataset
        grid_disp_range = cars_dataset.CarsDataset(
            "arrays", name="grid_disp_range_unknown_pair"
        )
        global_infos_cars_ds = cars_dataset.CarsDataset("dict")

        # Generate profile
        raster_profile = DefaultGTiffProfile(count=1)

        # saving infos
        # disp grids

        if self.save_intermediate_data:
            grid_min_path = os.path.join(pair_folder, "disp_min_grid.tif")
            grid_max_path = os.path.join(pair_folder, "disp_max_grid.tif")
            safe_makedirs(pair_folder)
        else:
            if pair_folder is None:
                tmp_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            else:
                tmp_folder = os.path.join(pair_folder, "tmp")
            safe_makedirs(tmp_folder)
            self.orchestrator.add_to_clean(tmp_folder)
            grid_min_path = os.path.join(tmp_folder, "disp_min_grid.tif")
            grid_max_path = os.path.join(tmp_folder, "disp_max_grid.tif")

        if None not in (dmin, dmax):
            # use global disparity range
            if None not in (dem_min, dem_max):
                raise RuntimeError("Mix between local and global mode")

            # Only one tile
            grid_disp_range.tiling_grid = np.array([[[0, nb_rows, 0, nb_cols]]])

        elif None not in (dem_min, dem_max):

            # Generate multiple tiles
            grid_tile_size = self.epi_disp_grid_tile_size
            grid_disp_range.tiling_grid = tiling.generate_tiling_grid(
                0,
                0,
                nb_rows,
                nb_cols,
                grid_tile_size,
                grid_tile_size,
            )

        # add tiling of  global_infos_cars_ds
        global_infos_cars_ds.tiling_grid = grid_disp_range.tiling_grid
        self.orchestrator.add_to_replace_lists(
            global_infos_cars_ds,
            cars_ds_name="global infos",
        )

        self.orchestrator.add_to_save_lists(
            grid_min_path,
            dm_cst.DISP_MIN_GRID,
            grid_disp_range,
            dtype=np.float32,
            nodata=0,
            cars_ds_name="disp_min_grid",
        )

        self.orchestrator.add_to_save_lists(
            grid_max_path,
            dm_cst.DISP_MAX_GRID,
            grid_disp_range,
            dtype=np.float32,
            nodata=0,
            cars_ds_name="disp_max_grid",
        )
        [saving_info] = (  # pylint: disable=unbalanced-tuple-unpacking
            self.orchestrator.get_saving_infos([grid_disp_range])
        )
        [saving_info_global_infos] = self.orchestrator.get_saving_infos(
            [global_infos_cars_ds]
        )

        # Generate grids on dict format
        grid_disp_range_dict = {
            "grid_min_path": grid_min_path,
            "grid_max_path": grid_max_path,
            "global_min": None,
            "global_max": None,
            "step_row": step_row,
            "step_col": step_col,
            "row_range": row_range,
            "col_range": col_range,
        }

        if None not in (dmin, dmax):
            # use global disparity range
            if None not in (dem_min, dem_max):
                raise RuntimeError("Mix between local and global mode")

            saving_info_global_infos_full = ocht.update_saving_infos(
                saving_info_global_infos, row=0, col=0
            )
            saving_info_full = ocht.update_saving_infos(
                saving_info, row=0, col=0
            )

            (
                grid_disp_range[0, 0],
                global_infos_cars_ds[0, 0],
            ) = self.orchestrator.cluster.create_task(
                generate_disp_range_const_tile_wrapper, nout=2
            )(
                row_range,
                col_range,
                dmin,
                dmax,
                raster_profile,
                saving_info_full,
                saving_info_global_infos_full,
            )

        elif None not in (dem_min, dem_max):

            # use filter to propagate min and max
            filter_overlap = (
                2
                * int(
                    self.disp_range_propagation_filter_size
                    / self.local_disp_grid_step
                )
                + 1
            )

            for col in range(grid_disp_range.shape[1]):
                for row in range(grid_disp_range.shape[0]):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    saving_info_global_infos_full = ocht.update_saving_infos(
                        saving_info_global_infos, row=row, col=col
                    )
                    array_window = grid_disp_range.get_window_as_dict(row, col)
                    (
                        grid_disp_range[row, col],
                        global_infos_cars_ds[row, col],
                    ) = self.orchestrator.cluster.create_task(
                        generate_disp_range_from_dem_wrapper, nout=2
                    )(
                        array_window,
                        row_range,
                        col_range,
                        sensor_image_right,
                        grid_right,
                        geom_plugin_with_dem_and_geoid,
                        dem_min,
                        dem_max,
                        raster_profile,
                        full_saving_info,
                        saving_info_global_infos_full,
                        filter_overlap,
                        disp_to_alt_ratio,
                        disp_min_threshold=self.disp_min_threshold,
                        disp_max_threshold=self.disp_max_threshold,
                    )

        # Compute grid
        self.orchestrator.breakpoint()

        # Fill global infos
        mins, maxs = [], []
        for row in range(global_infos_cars_ds.shape[0]):
            for col in range(global_infos_cars_ds.shape[1]):
                try:
                    dict_data = global_infos_cars_ds[row, col].data
                    mins.append(dict_data["global_min"])
                    maxs.append(dict_data["global_max"])
                except Exception:
                    logging.info(
                        "Tile {} {} not computed in epi disp range"
                        " generation".format(row, col)
                    )
        grid_disp_range_dict["global_min"] = np.floor(np.nanmin(mins))
        grid_disp_range_dict["global_max"] = np.ceil(np.nanmax(maxs))

        return grid_disp_range_dict

    def run(  # pylint: disable=too-many-positional-arguments
        self,
        epipolar_images_left,
        epipolar_images_right,
        local_tile_optimal_size_fun,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        disp_range_grid=None,
        compute_disparity_masks=False,
        margins_to_keep=0,
        texture_bands=None,
        classif_bands_to_mask=None,
    ):
        """
        Run Matching application.

        Create CarsDataset filled with xarray.Dataset, corresponding
        to epipolar disparities, on the same geometry than
        epipolar_images_left.

        :param epipolar_images_left: tiled left epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled right epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "texture"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"
                        "transform", "crs", "valid_pixels", "no_data_mask",
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size"
        :type epipolar_images_right: CarsDataset
        :param local_tile_optimal_size_fun: function to compute local
            optimal tile size
        :type local_tile_optimal_size_fun: func
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str
        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param disp_to_alt_ratio: disp to alti ratio used for performance map
        :type disp_to_alt_ratio: float
        :param margins_to_keep: margin to keep after dense matching
        :type margins_to_keep: int
        :param texture_bands: indices of bands from epipolar_images_left
            used for output texture
        :type texture_bands: list

        :return: disparity map: \
            The CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                 "disp_min_tiling", "disp_max_tiling"

        :rtype: CarsDataset
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

        if epipolar_images_left.dataset_type == "arrays":
            # Create CarsDataset
            # Epipolar_disparity
            epipolar_disparity_map = cars_dataset.CarsDataset(
                "arrays", name="dense_matching_" + pair_key
            )
            epipolar_disparity_map.create_empty_copy(epipolar_images_left)
            # Modify overlaps
            epipolar_disparity_map.overlaps = (
                format_transformation.reduce_overlap(
                    epipolar_disparity_map.overlaps, margins_to_keep
                )
            )

            # Update attributes to get epipolar info
            epipolar_disparity_map.attributes.update(
                epipolar_images_left.attributes
            )

            # Save disparity maps
            if self.save_intermediate_data:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp.tif"),
                    cst_disp.MAP,
                    epipolar_disparity_map,
                    cars_ds_name="epi_disp",
                    nodata=-9999,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_image.tif"),
                    cst.EPI_TEXTURE,
                    epipolar_disparity_map,
                    cars_ds_name="epi_disp_color",
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_mask.tif"),
                    cst_disp.VALID,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_mask",
                    optional_data=True,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_disp_classif.tif"),
                    cst.EPI_CLASSIFICATION,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_classif",
                    optional_data=True,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_confidence.tif",
                    ),
                    cst_disp.CONFIDENCE,
                    epipolar_disparity_map,
                    cars_ds_name="confidence",
                    optional_data=True,
                )

                # disparity grids
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_min.tif",
                    ),
                    cst_disp.EPI_DISP_MIN_GRID,
                    epipolar_disparity_map,
                    cars_ds_name="disp_min",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_max.tif",
                    ),
                    cst_disp.EPI_DISP_MAX_GRID,
                    epipolar_disparity_map,
                    cars_ds_name="disp_max",
                )
                self.orchestrator.add_to_save_lists(
                    os.path.join(
                        pair_folder,
                        "epi_disp_filling.tif",
                    ),
                    cst_disp.FILLING,
                    epipolar_disparity_map,
                    dtype=np.uint8,
                    cars_ds_name="epi_disp_filling",
                    nodata=255,
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [epipolar_disparity_map]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    dm_cst.DENSE_MATCHING_RUN_TAG: {
                        pair_key: {
                            "global_disp_min": disp_range_grid["global_min"],
                            "global_disp_max": disp_range_grid["global_max"],
                        },
                    },
                }
            }
            self.orchestrator.update_out_info(updating_dict)
            logging.info(
                "Compute disparity: number tiles: {}".format(
                    epipolar_disparity_map.shape[1]
                    * epipolar_disparity_map.shape[0]
                )
            )

            nb_total_tiles_roi = 0

            # broadcast grids
            # Transform grids to CarsDict for broadcasting
            # due to Dask issue https://github.com/dask/dask/issues/9969
            broadcasted_disp_range_grid = self.orchestrator.cluster.scatter(
                CarsDict(disp_range_grid)
            )

            # Generate disparity maps
            for col in range(epipolar_disparity_map.shape[1]):
                for row in range(epipolar_disparity_map.shape[0]):
                    use_tile = False
                    crop_with_range = None
                    if type(None) not in (
                        type(epipolar_images_left[row, col]),
                        type(epipolar_images_right[row, col]),
                    ):
                        use_tile = True
                        nb_total_tiles_roi += 1

                        # Compute optimal tile size for tile
                        (
                            _,
                            _,
                            crop_with_range,
                        ) = local_tile_optimal_size_fun(
                            np.array(
                                epipolar_images_left.attributes[
                                    "disp_min_tiling"
                                ]
                            )[row, col],
                            np.array(
                                epipolar_images_left.attributes[
                                    "disp_max_tiling"
                                ]
                            )[row, col],
                        )

                    if use_tile:
                        # update saving infos  for potential replacement
                        full_saving_info = ocht.update_saving_infos(
                            saving_info, row=row, col=col
                        )
                        # Compute disparity
                        (
                            epipolar_disparity_map[row, col]
                        ) = self.orchestrator.cluster.create_task(
                            basic_dense_matching_wrapper
                        )(
                            self.dense_matching_method,
                            epipolar_images_left[row, col],
                            epipolar_images_right[row, col],
                            broadcasted_disp_range_grid,
                            saving_info=full_saving_info,
                            compute_disparity_masks=compute_disparity_masks,
                            crop_with_range=crop_with_range,
                            left_overlaps=cars_dataset.overlap_array_to_dict(
                                epipolar_disparity_map.overlaps[row, col]
                            ),
                            margins_to_keep=margins_to_keep,
                            texture_bands=texture_bands,
                            classif_bands_to_mask=classif_bands_to_mask,
                        )

        else:
            logging.error(
                "DenseMatching application doesn't "
                "support this input data format"
            )
        return epipolar_disparity_map


def basic_dense_matching_wrapper(
    dense_matching_method,
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    disp_range_grid,
    saving_info=None,
    compute_disparity_masks=False,
    crop_with_range=None,
    left_overlaps=None,
    margins_to_keep=0,
    texture_bands=None,
    classif_bands_to_mask=None,
):
    """
    Matching application wrapper.

    Create CarsDataset filled with xarray.Dataset, corresponding
    to epipolar disparities, on the same geometry than
    left_image_object.

    :param left_image_object: tiled left epipolar CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data with keys : "im", "msk", "texture"
                - attrs with keys: "margins" with "disp_min" and "disp_max"\
                    "transform", "crs", "valid_pixels", "no_data_mask",\
                    "no_data_img"
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size"
    :type left_image_object: CarsDataset
    :param right_image_object: tiled right epipolar CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data with keys : "im", "msk", "texture"
                - attrs with keys: "margins" with "disp_min" and "disp_max"
                    "transform", "crs", "valid_pixels", "no_data_mask",
                    "no_data_img"
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size"
    :type right_image_object: CarsDataset
    :param disp_range_grid: minimum and maximum disparity grid
    :type disp_range_grid: CarsDict
    :param saving_info: information required to save output data
    :type saving_info: dict
    :param compute_disparity_masks: activate computation of disparity masks
    :type compute_disparity_masks: bool
    :param crop_with_range: crop disparity map using provided disparity range
    :type crop_with_range: int
    :param left_overlaps: overlaps associated to left image tiles
    :type left_overlaps: dict
    :param margins_to_keep: margin to keep after dense matching
    :type margins_to_keep: int
    :param texture_bands: indices of bands from epipolar_images_left
        used for output texture
    :type texture_bands: list
    :param classif_bands_to_mask: bands from classif to mask
    :type classif_bands_to_mask: list of str / int

    :return: Left to right disparity dataset
        Returned dataset is composed of :

        - cst_disp.MAP
        - cst_disp.VALID
        - cst.EPI_TEXTURE
    """
    return dense_matching_method.run(
        left_image_object,
        right_image_object,
        disp_range_grid,
        saving_info,
        compute_disparity_masks,
        crop_with_range,
        left_overlaps,
        margins_to_keep,
        texture_bands,
        classif_bands_to_mask,
    )

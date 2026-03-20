"""
This module contains the Pandora dense matching method implementation
"""

import collections

# Standard imports
import copy
import logging
import math

# Third party imports
import numpy as np
import pandora
import xarray as xr
from json_checker import And, Checker, Or
from pandora.check_configuration import check_pipeline_section
from pandora.img_tools import add_global_disparity
from pandora.state_machine import PandoraMachine
from scipy.ndimage import maximum_filter, minimum_filter

# CARS imports
from cars.applications.dense_matching import dense_matching_algo as dm_algo
from cars.applications.dense_matching import (
    dense_matching_wrappers as dm_wrappers,
)
from cars.applications.dense_matching.loaders.pandora_loader import (
    PandoraLoader,
)
from cars.applications.dense_matching.methods import (
    abstract_dense_matching_method as adm,
)
from cars.core import constants as cst
from cars.core import inputs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile

AbstractDenseMatchingMethod = adm.AbstractDenseMatchingMethod

# pylint:disable=C0302,R0902,R0917


def is_valid_perf_method(x):
    return all(y in ["risk", "intervals"] for y in x)


class PandoraMethod(
    AbstractDenseMatchingMethod,
    short_name=[
        "pandora_custom",
        "pandora_mccnn_sgm",
        "pandora_census_sgm_urban",
        "pandora_census_sgm_shadow",
        "pandora_census_sgm_mountain_and_vegetation",
        "pandora_census_sgm_homogeneous",
        "pandora_census_sgm_default",
        "pandora_census_sgm_sparse",
        "pandora_auto",
    ],
):
    """
    The implementation of Pandora as a Dense matching method
    """

    def __init__(self, conf):

        super().__init__(conf=conf)

        # non blocking check conf
        self.schema = {
            "method": str,
            "generate_ambiguity": bool,
            "performance_map_method": And(
                list,
                is_valid_perf_method,
            ),
            "perf_eta_max_ambiguity": float,
            "perf_eta_max_risk": float,
            "perf_eta_step": float,
            "perf_ambiguity_threshold": float,
            "classification_fusion_margin": int,
            "use_cross_validation": Or(bool, str),
            "denoise_disparity_map": bool,
            "used_band": str,
            "loader_conf": Or(dict, collections.OrderedDict, str, None),
            "loader": str,
            "confidence_filtering": dict,
            "threshold_disp_range_to_borders": bool,
            "filter_incomplete_disparity_range": bool,
            "edges_3sgm": bool,
        }

        # those will be defined in check_conf
        self.loader = None
        self.margins = None
        self.corr_config = None

        self.used_config = self.check_conf(conf)

        self.method = self.used_config["method"]
        self.generate_ambiguity = self.used_config["generate_ambiguity"]
        self.performance_map_method = self.used_config["performance_map_method"]
        self.perf_eta_max_ambiguity = self.used_config["perf_eta_max_ambiguity"]
        self.perf_eta_max_risk = self.used_config["perf_eta_max_risk"]
        self.perf_eta_step = self.used_config["perf_eta_step"]
        self.perf_ambiguity_threshold = self.used_config[
            "perf_ambiguity_threshold"
        ]
        self.classification_fusion_margin = self.used_config[
            "classification_fusion_margin"
        ]
        self.use_cross_validation = self.used_config["use_cross_validation"]
        self.denoise_disparity_map = self.used_config["denoise_disparity_map"]
        self.used_band = self.used_config["used_band"]
        self.loader_conf = self.used_config["loader_conf"]
        self.confidence_filtering = self.used_config["confidence_filtering"]
        self.threshold_disp_range_to_borders = self.used_config[
            "threshold_disp_range_to_borders"
        ]
        self.filter_incomplete_disparity_range = self.used_config[
            "filter_incomplete_disparity_range"
        ]
        self.edges_3sgm = self.used_config["edges_3sgm"]

    def check_conf(self, conf):
        """
        Merge user configuration with default values and validate schema.
        Extra keys in conf do not raise errors.
        """

        # Initialize conf if None
        if conf is None:
            conf = {}

        save_intermediate_data = conf.get("save_intermediate_data", False)

        default_perf_map_method = "risk" if save_intermediate_data else None

        # Default configuration
        default_conf = {
            "save_intermediate_data": False,
            "generate_ambiguity": save_intermediate_data,
            "performance_map_method": default_perf_map_method,
            "perf_eta_max_ambiguity": 0.99,
            "perf_eta_max_risk": 0.25,
            "perf_eta_step": 0.04,
            "perf_ambiguity_threshold": 0.6,
            "classification_fusion_margin": 5,
            "use_cross_validation": True,
            "denoise_disparity_map": False,
            "used_band": "b0",
            "loader_conf": None,
            "loader": "pandora",
            "confidence_filtering": {},
            "threshold_disp_range_to_borders": False,
            "filter_incomplete_disparity_range": True,
            "edges_3sgm": True,
        }

        # Merge defaults with user conf
        used_conf = default_conf.copy()
        used_conf.update(conf)

        # --- Update parameters

        if used_conf["use_cross_validation"] is True:
            used_conf["use_cross_validation"] = "fast"

        # Get/update perf map method as list
        perf_map_method = used_conf["performance_map_method"]
        if isinstance(perf_map_method, str):
            used_conf["performance_map_method"] = [perf_map_method]
        elif perf_map_method is None:
            used_conf["performance_map_method"] = []
        perf_map_method = used_conf["performance_map_method"]

        # Loader initialization
        # this overrides conf[loader], and sets:
        # self.loader
        # self.margins
        # self.corr_config
        self.check_conf_pandora_loader(used_conf, perf_map_method)

        # Validate only keys defined in schema
        conf_to_check = {k: used_conf[k] for k in self.schema if k in used_conf}

        checker = Checker(self.schema)
        checker.validate(conf_to_check)

        # additional checks: confidence_filtering
        conf_to_check["confidence_filtering"] = (
            self.check_conf_confidence_filtering(
                conf_to_check["confidence_filtering"]
            )
        )

        # return the conf without unvalidated keys
        return conf_to_check

    def check_conf_pandora_loader(self, conf, perf_map_method):
        """
        Check the pandora loader conf
        """

        loader = conf.get("loader")
        loader_conf = conf.get("loader_conf")

        default_method = "pandora_custom" if loader_conf else "pandora_auto"
        method = conf.get("method", default_method)
        if method == "pandora_auto" and loader_conf:
            raise RuntimeError(
                "Cannot use 'pandora_auto' method with a custom loader "
                "configuration"
            )

        logger = logging.getLogger("transitions.core")
        logger.addFilter(
            lambda record: "to model due to model override policy"
            not in record.getMessage()
        )
        pandora_loader = PandoraLoader(
            conf=loader_conf,
            method_name=method,
            generate_performance_map_from_risk="risk" in perf_map_method,
            generate_performance_map_from_intervals="intervals"
            in perf_map_method,
            generate_ambiguity=conf["generate_ambiguity"],
            perf_eta_max_ambiguity=conf["perf_eta_max_ambiguity"],
            perf_eta_max_risk=conf["perf_eta_max_risk"],
            perf_eta_step=conf["perf_eta_step"],
            use_cross_validation=conf["use_cross_validation"],
            denoise_disparity_map=conf["denoise_disparity_map"],
            used_band=conf["used_band"],
        )

        self.loader = pandora_loader
        self.corr_config = collections.OrderedDict(pandora_loader.get_conf())
        conf["loader"] = loader

        # create the dataset
        classif_bands = pandora_loader.get_classif_bands()
        fake_dataset = xr.Dataset(
            data_vars={
                "image": (["row", "col"], np.zeros((10, 10))),
                "classif": (
                    ["row", "col", "band_classif"],
                    np.zeros((10, 10, len(classif_bands)), dtype=np.int32),
                ),
            },
            coords={
                "band_im": [conf["used_band"]],
                "band_classif": classif_bands,
                "row": np.arange(10),
                "col": np.arange(10),
            },
            attrs={"disparity_source": [-1, 1]},
        )

        # Import plugins before checking configuration
        pandora.import_plugin()
        pandora_machine = PandoraMachine()

        corr_config_pipeline = {"pipeline": dict(self.corr_config["pipeline"])}

        saved_schema = copy.deepcopy(
            pandora.matching_cost.matching_cost.AbstractMatchingCost.schema
        )
        _ = check_pipeline_section(
            corr_config_pipeline, fake_dataset, fake_dataset, pandora_machine
        )
        # quick fix to remove when the problem is solved in pandora
        pandora.matching_cost.matching_cost.AbstractMatchingCost.schema = (
            saved_schema
        )
        self.margins = pandora_machine.margins.global_margins

    def check_conf_confidence_filtering(self, overloaded_conf):
        """
        Check the confidence filtering conf
        """

        default_conf = {
            "activated": True,
            "bounds_ratio_threshold": 0.2,
            "risk_ratio_threshold": 0.75,
            "bounds_range_threshold": 3,
            "risk_range_threshold": 9,
            "nan_threshold": 0.2,
            "win_nanratio": 20,
        }

        confidence_filtering_schema = {
            "activated": bool,
            "bounds_ratio_threshold": float,
            "risk_ratio_threshold": float,
            "bounds_range_threshold": int,
            "risk_range_threshold": int,
            "nan_threshold": float,
            "win_nanratio": int,
        }

        used_conf = default_conf.copy()
        used_conf.update(overloaded_conf)

        checker_confidence_filtering_schema = Checker(
            confidence_filtering_schema
        )
        checker_confidence_filtering_schema.validate(used_conf)

        return used_conf

    def get_method(self):
        """
        Returns the method used
        """
        return self.method

    def get_performance_map_parameters(self):
        """
        Get parameter linked to performance, that will be used in triangulation

        :return: parameters to use
        :type: dict
        """

        return {
            "performance_map_method": self.used_config[
                "performance_map_method"
            ],
            "perf_ambiguity_threshold": self.used_config[
                "perf_ambiguity_threshold"
            ],
        }

    @cars_profile(name="Get margin fun")
    def get_margins_fun(
        self,
        grid_left,
        disp_range_grid,
        min_elevation_offset,
        max_elevation_offset,
    ):
        """
        Get Margins function that generates margins needed by
        matching method, to use during resampling.

        :param grid_left: left epipolar grid
        :type grid_left: dict
        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param min_elevation_offset: minimum elevation offset
        :type min_elevation_offset: float
        :param max_elevation_offset: maximum elevation offset
        :type max_elevation_offset: float
        :return: function that generates margin for given roi
        :rtype: callable
        """

        disp_min_grid_arr, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_min_path"]
        )
        disp_max_grid_arr, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_max_path"]
        )
        step_row = disp_range_grid["step_row"]
        step_col = disp_range_grid["step_col"]
        row_range = disp_range_grid["row_range"]
        col_range = disp_range_grid["col_range"]

        # get disp_to_alt_ratio
        disp_to_alt_ratio = grid_left["disp_to_alt_ratio"]

        # Check if we need to override disp_min
        if min_elevation_offset is not None:
            user_disp_min = min_elevation_offset / disp_to_alt_ratio
            if np.any(disp_min_grid_arr < user_disp_min):
                logging.warning(
                    (
                        "Overridden disparity minimum "
                        "= {:.3f} pix. (= {:.3f} m.) "
                        "is greater than disparity minimum estimated "
                        "in prepare step "
                        "for current pair"
                    ).format(
                        user_disp_min,
                        min_elevation_offset,
                    )
                )
            disp_min_grid_arr[:, :] = user_disp_min

        # Check if we need to override disp_max
        if max_elevation_offset is not None:
            user_disp_max = max_elevation_offset / disp_to_alt_ratio
            if np.any(disp_max_grid_arr > user_disp_max):
                logging.warning(
                    (
                        "Overridden disparity maximum "
                        "= {:.3f} pix. (or {:.3f} m.) "
                        "is lower than disparity maximum estimated "
                        "in prepare step "
                        "for current pair"
                    ).format(
                        user_disp_max,
                        max_elevation_offset,
                    )
                )
            disp_max_grid_arr[:, :] = user_disp_max

        # Compute global range of logging
        disp_min_global = np.min(disp_min_grid_arr)
        disp_max_global = np.max(disp_max_grid_arr)

        logging.info(
            "Global Disparity range for current pair:  "
            "[{:.3f} pix., {:.3f} pix.] "
            "(or [{:.3f} m., {:.3f} m.])".format(
                disp_min_global,
                disp_max_global,
                disp_min_global * disp_to_alt_ratio,
                disp_max_global * disp_to_alt_ratio,
            )
        )

        def margins_wrapper(row_min, row_max, col_min, col_max):
            """
            Generates margins Dataset used in resampling

            :param row_min: row min
            :param row_max: row max
            :param col_min: col min
            :param col_max: col max

            :return: margins
            :rtype: xr.Dataset
            """

            assert row_min < row_max
            assert col_min < col_max

            # Get region in grid

            grid_row_min = max(0, int(np.floor((row_min - 1) / step_row)) - 1)
            grid_row_max = min(
                len(row_range), int(np.ceil((row_max + 1) / step_row) + 1)
            )
            grid_col_min = max(0, int(np.floor((col_min - 1) / step_col)) - 1)
            grid_col_max = min(
                len(col_range), int(np.ceil((col_max + 1) / step_col)) + 1
            )

            # Compute disp min and max in row
            disp_min = np.min(
                disp_min_grid_arr[
                    grid_row_min:grid_row_max, grid_col_min:grid_col_max
                ]
            )
            disp_max = np.max(
                disp_max_grid_arr[
                    grid_row_min:grid_row_max, grid_col_min:grid_col_max
                ]
            )
            # round disp min and max
            disp_min = int(math.floor(disp_min))
            disp_max = int(math.ceil(disp_max))

            # Compute margins for the correlator
            margins = dm_wrappers.get_margins(self.margins, disp_min, disp_max)

            return margins

        return margins_wrapper

    @cars_profile(name="Optimal size estimation")
    def get_optimal_tile_size(
        self,
        disp_range_grid,
        max_ram_per_worker,
        min_epi_tile_size,
        max_epi_tile_size,
        local_disp_grid_step,
        epipolar_tile_margin_in_percent,
    ):
        """
        Get the optimal tile size to use during dense matching.

        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param max_ram_per_worker: maximum RAM per worker
        :type max_ram_per_worker: int
        :param min_epi_tile_size: minimum epipolar tile size
        :type min_epi_tile_size: int
        :param max_epi_tile_size: maximum epipolar tile size
        :type max_epi_tile_size: int
        :param local_disp_grid_step: disparity grid step
        :type local_disp_grid_step: int
        :param epipolar_tile_margin_in_percent: margin percentage
        :type epipolar_tile_margin_in_percent: float
        :return: optimal tile size and local function
        :rtype: tuple
        """

        disp_min_grids, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_min_path"]
        )
        disp_max_grids, _ = inputs.rasterio_read_as_array(
            disp_range_grid["grid_max_path"]
        )

        # use max tile size as overlap for min and max:
        # max Point to point diff is less than diff of tile

        # use filter of size max_epi_tile_size
        overlap = 3 * int(max_epi_tile_size / local_disp_grid_step)

        disp_min_grids = minimum_filter(
            disp_min_grids, size=[overlap, overlap], mode="nearest"
        )
        disp_max_grids = maximum_filter(
            disp_max_grids, size=[overlap, overlap], mode="nearest"
        )

        # Worst cases scenario:
        # 1: [global max - max diff, global max]
        # 2: [global min, global min  max diff]

        max_diff = np.round(np.nanmax(disp_max_grids - disp_min_grids)) + 1
        global_min = np.floor(np.nanmin(disp_min_grids))
        global_max = np.ceil(np.nanmax(disp_max_grids))

        # Get tiling param
        opt_epipolar_tile_size_1 = (
            dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                global_min,
                global_min + max_diff,
                min_epi_tile_size,
                max_epi_tile_size,
                max_ram_per_worker,
                margin=epipolar_tile_margin_in_percent,
            )
        )
        opt_epipolar_tile_size_2 = (
            dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                global_max - max_diff,
                global_max,
                min_epi_tile_size,
                max_epi_tile_size,
                max_ram_per_worker,
                margin=epipolar_tile_margin_in_percent,
            )
        )

        # return worst case
        opt_epipolar_tile_size = min(
            opt_epipolar_tile_size_1, opt_epipolar_tile_size_2
        )

        # Define function to compute local optimal size for each tile
        def local_tile_optimal_size_fun(local_disp_min, local_disp_max):
            """
            Compute optimal tile size for tile

            :return: local tile size, global optimal tile sizes

            """
            local_opt_tile_size = (
                dm_wrappers.optimal_tile_size_pandora_plugin_libsgm(
                    local_disp_min,
                    local_disp_max,
                    0,
                    20000,  # arbitrary
                    max_ram_per_worker,
                    margin=epipolar_tile_margin_in_percent,
                )
            )

            # Get max range to use with current optimal size
            max_range = dm_wrappers.get_max_disp_from_opt_tile_size(
                opt_epipolar_tile_size,
                max_ram_per_worker,
                margin=epipolar_tile_margin_in_percent,
                used_disparity_range=(local_disp_max - local_disp_min),
            )

            return local_opt_tile_size, opt_epipolar_tile_size, max_range

        return opt_epipolar_tile_size, local_tile_optimal_size_fun

    def run(
        self,
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
        Run dense matching on a pair of epipolar images.

        Compute the disparity map between left and right images using the
        configured dense matching method.

        :param left_image_object: left epipolar image as xarray Dataset with:
                - data with keys: "im", optionally "msk", "texture", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max",
                "transform", "crs", "valid_pixels", "no_data_mask",
                "no_data_img"
        :type left_image_object: xr.Dataset
        :param right_image_object: right epipolar image as xarray Dataset with:
                - data with keys: "im", optionally "msk", "texture", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max",
                "transform", "crs", "valid_pixels", "no_data_mask",
                "no_data_img"
        :type right_image_object: xr.Dataset
        :param disp_range_grid: minimum and maximum disparity grid
        :type disp_range_grid: dict
        :param saving_info: information for saving outputs
        :type saving_info: dict
        :param compute_disparity_masks: activate computation of disparity masks
        :type compute_disparity_masks: bool
        :param crop_with_range: crop disparity map using provided range
        :type crop_with_range: int or None
        :param left_overlaps: overlaps associated to left image tiles
        :type left_overlaps: dict or None
        :param margins_to_keep: margins to keep after dense matching
        :type margins_to_keep: int
        :param texture_bands: indices of bands from left image used for texture
        :type texture_bands: list
        :param classif_bands_to_mask: bands from classification to mask
        :type classif_bands_to_mask: list of str or int

        :return: disparity map.

            The CarsDataset contains:

            - N x M delayed tiles.

            Each tile is an xarray Dataset containing:

            - data with keys: "disp", "disp_msk"
            - attrs with keys: "profile", "window", "overlaps"

            - attributes containing:

            - "largest_epipolar_region"
            - "opt_epipolar_tile_size"
            - "disp_min_tiling"
            - "disp_max_tiling"

        :rtype: CarsDataset
        """
        if self.edges_3sgm and "edges_mask" in left_image_object:
            self.corr_config["pipeline"]["optimization"][
                "optimization_method"
            ] = "3sgm"
            self.corr_config["pipeline"]["optimization"]["geometric_prior"] = {
                "source": "edges"
            }
            # convert to 2D array for sgm
            left_image_object["edges"] = left_image_object[
                "edges_mask"
            ].squeeze()

        # transform disp_range_grid back to dict
        disp_range_grid = disp_range_grid.data
        # Generate disparity grids
        (
            disp_min_grid,
            disp_max_grid,
        ) = dm_algo.compute_disparity_grid(
            disp_range_grid,
            left_image_object,
            right_image_object,
            self.used_band,
            self.threshold_disp_range_to_borders,
        )

        global_disp_min = disp_range_grid["global_min"]
        global_disp_max = disp_range_grid["global_max"]

        # add global disparity in case of ambiguity normalization
        left_image_object = add_global_disparity(
            left_image_object, global_disp_min, global_disp_max
        )

        # Crop interval if needed
        mask_crop = np.zeros(disp_min_grid.shape, dtype=int)
        is_cropped = False
        if crop_with_range is not None:
            current_min = np.min(disp_min_grid)
            current_max = np.max(disp_max_grid)
            if (current_max - current_min) > crop_with_range:
                is_cropped = True
                logging.warning("disparity range for current tile is cropped")
                # crop
                new_min = (
                    current_min * crop_with_range / (current_max - current_min)
                )
                new_max = (
                    current_max * crop_with_range / (current_max - current_min)
                )

                mask_crop = np.logical_or(
                    disp_min_grid < new_min, disp_max_grid > new_max
                )
                mask_crop = mask_crop.astype(bool)
                disp_min_grid[mask_crop] = new_min
                disp_max_grid[mask_crop] = new_max

        # Compute disparity
        # TODO : remove overwriting of EPI_MSK
        disp_dataset = dm_algo.compute_disparity(
            left_image_object,
            right_image_object,
            self.corr_config,
            self.used_band,
            disp_min_grid=disp_min_grid,
            disp_max_grid=disp_max_grid,
            compute_disparity_masks=compute_disparity_masks,
            cropped_range=mask_crop,
            margins_to_keep=margins_to_keep,
            classification_fusion_margin=self.classification_fusion_margin,
            texture_bands=texture_bands,
            filter_incomplete_disparity_range=(
                self.filter_incomplete_disparity_range
            ),
            classif_bands_to_mask=classif_bands_to_mask,
        )

        mask = disp_dataset["disp_msk"].values
        disp_map = disp_dataset["disp"].values
        disp_map[mask == 0] = np.nan

        # Filtering by using the confidence
        requested_confidence = [
            "confidence_from_risk_min.cars_2",
            "confidence_from_risk_max.cars_2",
            "confidence_from_interval_bounds_inf.cars_3",
            "confidence_from_interval_bounds_sup.cars_3",
        ]

        if (
            all(key in disp_dataset for key in requested_confidence)
            and self.confidence_filtering["activated"] is True
        ):
            dm_wrappers.confidence_filtering(
                disp_dataset,
                requested_confidence,
                self.confidence_filtering,
            )

        # Fill with attributes
        cars_dataset.fill_dataset(
            disp_dataset,
            saving_info=saving_info,
            window=cars_dataset.get_window_dataset(left_image_object),
            profile=cars_dataset.get_profile_rasterio(left_image_object),
            attributes={cst.CROPPED_DISPARITY_RANGE: is_cropped},
            overlaps=left_overlaps,
        )

        return disp_dataset

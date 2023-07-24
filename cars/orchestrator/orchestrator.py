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
this module contains the orchestrator class
"""

# Standard imports
import collections
import logging
import os
import shutil
import sys

# Third party imports
import tempfile
import traceback

from tqdm import tqdm

# CARS imports
from cars.core.cars_logging import add_progress_message
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.abstract_cluster import AbstractCluster
from cars.orchestrator.orchestrator_constants import CARS_DS_COL, CARS_DS_ROW
from cars.orchestrator.registry import id_generator as id_gen
from cars.orchestrator.registry import replacer_registry, saver_registry


class Orchestrator:
    """
    Orchestrator
    """

    # pylint: disable=too-many-instance-attributes

    # flake8: noqa: C901
    def __init__(
        self,
        orchestrator_conf=None,
        out_dir=None,
        launch_worker=True,
        out_json_path=None,
    ):
        """
        Init function of Orchestrator.
        Creates Cluster and Registry for CarsDatasets

        :param orchestrator_conf: configuration of distribution

        """
        # init list of path to clean at the end
        self.tmp_dir_list = []

        # out_dir
        if out_dir is not None:
            self.out_dir = out_dir
        else:
            self.out_dir = tempfile.mkdtemp()
            self.add_to_clean(self.out_dir)
            logging.debug("No out_dir defined")

        self.launch_worker = launch_worker

        # overload orchestrator_conf
        if orchestrator_conf is None:
            orchestrator_conf = {"mode": "local_dask"}
            logging.info(
                "No orchestrator configuration given: "
                "local_dask mode is used"
            )

        # set OTB_MAX_RAM_HINT
        self.former_otb_max_ram = os.environ.get("OTB_MAX_RAM_HINT", None)
        os.environ["OTB_MAX_RAM_HINT"] = "3000"  # 60% 5Gb

        # init cluster
        self.cluster = AbstractCluster(  # pylint: disable=E0110
            orchestrator_conf, self.out_dir, launch_worker=self.launch_worker
        )
        self.conf = self.cluster.get_conf()

        # Init IdGenerator
        self.id_generator = id_gen.IdGenerator()
        # init CarsDataset savers registry
        self.cars_ds_savers_registry = saver_registry.CarsDatasetsRegistrySaver(
            self.id_generator
        )
        # init CarsDataset replacement registry
        self.cars_ds_replacer_registry = (
            replacer_registry.CarsDatasetRegistryReplacer(self.id_generator)
        )

        # init cars_ds_names_info for pbar printing
        self.cars_ds_names_info = []

        # outjson
        self.out_json_path = out_json_path
        if self.out_json_path is None:
            os.path.join(self.out_dir, "content.json")
        self.out_json = {}

    def add_to_clean(self, tmp_dir):
        self.tmp_dir_list.append(tmp_dir)

    def get_conf(self):
        """
        Get orchestrator conf

        :return: orchestrator conf
        """

        return self.conf

    def add_to_save_lists(
        self,
        file_name,
        tag,
        cars_ds,
        dtype="float32",
        nodata=0,
        cars_ds_name=None,
        optional_data=False,
    ):
        """
        Save file to list in order to be saved later

        :param file_name: file name
        :param tag: tag
        :param cars_ds: cars dataset to register
        :param cars_ds_name: name corresponding to CarsDataset,
          for information during logging
        :param optional_data: True if the data is optionnal
        :type optional_data: bool
        """

        self.cars_ds_savers_registry.add_file_to_save(
            file_name,
            cars_ds,
            tag=tag,
            dtype=dtype,
            nodata=nodata,
            optional_data=optional_data,
        )

        # add name if exists
        if cars_ds_name is not None:
            self.cars_ds_names_info.append(cars_ds_name)

    def add_to_replace_lists(self, cars_ds, cars_ds_name=None):
        """
        Add CarsDataset to replacing Registry

        :param cars_ds: CarsDataset to replace
        :type cars_ds: CarsDataset
        :param cars_ds_name: name corresponding to CarsDataset,
            for information during logging
        """

        self.cars_ds_replacer_registry.add_cars_ds_to_replace(cars_ds)

        # add name if exists
        if cars_ds_name is not None:
            self.cars_ds_names_info.append(cars_ds_name)

    def save_out_json(self):
        """
        Check out_json and save it to file
        """

        # TODO check schema ?

        # dump file
        if self.out_json_path is not None:
            cars_dataset.save_dict(
                self.out_json, self.out_json_path, safe_save=True
            )

    def update_out_info(self, new_dict):
        """
        Update self.out_json with new dict

        :param new_dict: dict to merge
        :type new_dict: dict
        """

        # TODO merge with safe creation of new keys of application
        # when 2 same applications are used

        def merge_dicts(dict1, dict2):
            """
            Merge dict2 into dict 1

            :param dict1: dict 1
            :type dict1: dict
            :param dict2: dict 2
            :type dict2: dict

            """

            for key, value2 in dict2.items():
                value1 = dict1.get(key)
                if isinstance(value1, collections.abc.Mapping) and isinstance(
                    value2, collections.abc.Mapping
                ):
                    merge_dicts(value1, value2)
                else:
                    dict1[key] = value2

        merge_dicts(self.out_json, new_dict)

    def get_saving_infos(self, cars_ds_list):
        """
        Get saving infos of given cars datasets

        :param cars_ds_list: list of cars datasets
        :type cars_ds_list: list[CarsDataset]

        :return : list of saving infos
        :rtype: list[dict]
        """

        saving_infos = []

        for cars_ds in cars_ds_list:
            saving_infos.append(self.id_generator.get_saving_infos(cars_ds))

        return saving_infos

    def compute_futures(self):
        """
        Compute all futures from regitries

        """

        # save json
        if self.launch_worker:
            self.save_out_json()

            # run compute and save files
            logging.info("Compute delayed ...")
            # Flatten to list
            delayed_objects = flatten_object(
                self.cars_ds_savers_registry.get_cars_datasets_list()
                + self.cars_ds_replacer_registry.get_cars_datasets_list()
            )

            # Compute delayed
            future_objects = self.cluster.start_tasks(delayed_objects)

            # Save objects when they are computed
            logging.info("Wait for futures results ...")
            add_progress_message(
                "Data list to process: [ {} ] ...".format(
                    " , ".join(list(set(self.cars_ds_names_info)))
                )
            )
            tqdm_message = "Tiles processing: "
            # if loglevel > PROGRESS level tqdm display the data list
            if logging.getLogger().getEffectiveLevel() > 21:
                tqdm_message = "Processing Tiles: [ {} ] ...".format(
                    " , ".join(list(set(self.cars_ds_names_info)))
                )
            pbar = tqdm(
                total=len(future_objects),
                desc=tqdm_message,
                position=0,
                leave=True,
                file=sys.stdout,
            )
            for future_obj in self.cluster.future_iterator(future_objects):
                # get corresponding CarsDataset and save tile
                if future_obj is not None:
                    # Save future if needs to
                    self.cars_ds_savers_registry.save(future_obj)
                    # Replace future in cars_ds if needs to
                    self.cars_ds_replacer_registry.replace(future_obj)
                else:
                    logging.debug("None tile: not saved")
                pbar.update()

            # close files
            logging.info("Close files ...")
            self.cars_ds_savers_registry.cleanup()
        else:
            logging.debug(
                "orchestrator launch_worker is False, no content.json saved"
            )

    def reset_registries(self):
        """
        Reset registries
        """
        # reset registries
        # CarsDataset savers registry
        self.cars_ds_savers_registry = saver_registry.CarsDatasetsRegistrySaver(
            self.id_generator
        )

        #  CarsDataset replacement registry
        self.cars_ds_replacer_registry = (
            replacer_registry.CarsDatasetRegistryReplacer(self.id_generator)
        )

        # reset cars_ds names infos
        self.cars_ds_names_info = []

    def breakpoint(self):
        """
        Breakpoint : compute all delayed, save and replace data
        in CarsDatasets

        """

        # Compute futures
        try:
            self.compute_futures()
        except Exception as exc:
            # reset registries
            self.reset_registries()
            raise RuntimeError(traceback.format_exc()) from exc

        # reset registries
        self.reset_registries()

    def __enter__(self):
        """
        Function run on enter

        """

        return self

    def __exit__(self, exc_type, exc_value, traceback_msg):
        """
        Function run on exit.

        Compute cluster tasks, save futures to be saved, and cleanup cluster
        and files

        """

        # Compute futures
        self.breakpoint()

        # save outjson
        # TODO

        # TODO : check_json

        # close cluster
        logging.info("Close cluster ...")
        if self.launch_worker:
            self.cluster.cleanup()

        # # clean tmp dir
        for tmp_dir in self.tmp_dir_list:
            if tmp_dir is not None and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        # reset OTB_MAX_RAM_HINT
        if self.former_otb_max_ram is None:
            del os.environ["OTB_MAX_RAM_HINT"]
        else:
            os.environ["OTB_MAX_RAM_HINT"] = self.former_otb_max_ram


def flatten_object(cars_ds_list):
    """
    Flatten list of CarsDatasets to list of delayed

    :param cars_ds_list: list of cars datasets flatten
    :type cars_ds_list: list[CarsDataset]

    :return: list of delayed
    :rtype: list[Delayed]
    """

    # remove duplicates
    cleaned_cars_ds_list = list(dict.fromkeys(cars_ds_list))

    # flatten datasets
    flattened_objects = []

    if len(cleaned_cars_ds_list) == 1 and cleaned_cars_ds_list[0] is None:
        return []

    # add obj flattened
    for cars_ds in cleaned_cars_ds_list:
        flattened_objects += [
            obj
            for obj_list in cars_ds.tiles
            for obj in obj_list
            if obj is not None
        ]

    return flattened_objects


def update_saving_infos(saving_info_left, row=None, col=None):
    """
    Update saving infos dict with row and col arguments

    :param saving_info_left: saving infos
    :type saving_info_left: dict
    :param row: row
    :type row: int
    :param col: col
    :type col: int

    :return: updated saving infos dict
    :rtype: dict
    """

    full_saving_infos = saving_info_left.copy()

    if row is not None:
        full_saving_infos[CARS_DS_ROW] = row

    if col is not None:
        full_saving_infos[CARS_DS_COL] = col

    return full_saving_infos

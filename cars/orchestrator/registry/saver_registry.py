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
This module contains the saver registry class
"""


# Standard imports
import logging
import os
import traceback

# CARS imports
from cars.orchestrator.registry.abstract_registry import (
    AbstractCarsDatasetRegistry,
)


class CarsDatasetsRegistrySaver(AbstractCarsDatasetRegistry):
    """
    CarsDatasetsRegistrySaver
    This registry manages the saving of arriving future results
    """

    def __init__(self, id_generator):
        """
        Init function of CarsDatasetsRegistrySaver

        :param id_generator: id generator
        :type id_generator: IdGenerator

        """
        super().__init__(id_generator)
        self.registered_cars_datasets_savers = []

    def get_cars_ds(self, future_result):
        """
        Get a list of registered CarsDataset

        :param obj: object to get cars dataset from

        :return corresponding CarsDataset
        :rtype: CarsDataset
        """

        obj_id = self.get_future_cars_dataset_id(future_result)
        cars_ds_saver = self.get_cars_ds_saver_corresponding_id(obj_id)
        if cars_ds_saver is None:
            return None
        return cars_ds_saver.cars_ds

    def get_cars_datasets_list(self):
        """
        Get a list of registered CarsDataset

        :return list of CarsDataset
        :rtype: list(CarsDataset)
        """

        cars_ds_list = []

        for cars_ds_saver in self.registered_cars_datasets_savers:
            cars_ds_list.append(cars_ds_saver.cars_ds)

        return cars_ds_list

    def cars_dataset_in_registry(self, cars_ds):
        """
        Check if a CarsDataset is already registered, return id if exists

        :param cars_ds: cars dataset
        :type cars_ds: CarsDataset

        :return : True if in registry, if of cars dataset
        :rtype : Tuple(bool, int)
        """

        in_registry = False
        registered_id = None
        for obj in self.registered_cars_datasets_savers:
            if cars_ds == obj.cars_ds:
                in_registry = True
                registered_id = obj.obj_id
                break

        return in_registry, registered_id

    def get_cars_ds_saver_corresponding_cars_dataset(self, cars_ds):
        """
        Get the SingleCarsDatasetSaver corresponding to given cars dataset

        :param cars_ds: cars dataset

        :return : single cars dataset saver
        :rtype : SingleCarsDatasetSaver
        """

        cars_ds_saver = None
        for obj in self.registered_cars_datasets_savers:
            if cars_ds == obj.cars_ds:
                cars_ds_saver = obj
                break

        return cars_ds_saver

    def get_cars_ds_saver_corresponding_id(self, obj_id):
        """
        Get the SingleCarsDatasetSaver corresponding to given id

        :param obj_id: cars dataset id
        :type obj_id: int

        :return : single cars dataset saver
        :rtype : SingleCarsDatasetSaver
        """

        cars_ds_saver = None
        for obj in self.registered_cars_datasets_savers:
            if obj_id == obj.obj_id:
                cars_ds_saver = obj
                break

        return cars_ds_saver

    def save(self, future_result):
        """
        Save future result

        :param future_result: xr.Dataset or pd.DataFrame

        """

        obj_id = self.get_future_cars_dataset_id(future_result)
        cars_ds_saver = self.get_cars_ds_saver_corresponding_id(obj_id)

        if cars_ds_saver is not None:
            # save
            if future_result is not None:
                cars_ds_saver.save(future_result)
            else:
                logging.debug("Future result tile is None -> not saved")

    def add_file_to_save(
        self,
        file_name,
        cars_ds,
        tag=None,
        dtype=None,
        nodata=None,
        optional_data=False,
        save_points_cloud_by_pair=False,
    ):
        """
        Add file corresponding to cars_dataset to registered_cars_datasets

        :param file_name: file name to save futures to
        :type file_name: str
        :param cars_ds: CarsDataset to register
        :type cars_ds: CarsDataset
        :param tag: tag to save
        :type tag: str
        :param dtype: dtype
        :type dtype: str
        :param nodata: no data value
        :type nodata: float
        :param optional_data: True if the data is optionnal
        :type optional_data: bool
        :param save_points_cloud_by_pair:
        :type save_points_cloud_by_pair: bool
        """

        if not self.cars_dataset_in_registry(cars_ds)[0]:
            # Generate_id
            new_id = self.id_generator.get_new_id(cars_ds)
            # create CarsDataset saver
            cars_ds_saver = SingleCarsDatasetSaver(new_id, cars_ds)
            # add to list
            self.registered_cars_datasets_savers.append(cars_ds_saver)
        else:
            cars_ds_saver = self.get_cars_ds_saver_corresponding_cars_dataset(
                cars_ds
            )

        # update cars_ds_saver
        cars_ds_saver.add_file(
            file_name,
            tag=tag,
            dtype=dtype,
            nodata=nodata,
            optional_data=optional_data,
            save_points_cloud_by_pair=save_points_cloud_by_pair,
        )

    def cleanup(self):
        """
        Cleanup function.

        Close correctly all opened files.

        """
        for obj in self.registered_cars_datasets_savers:
            obj.cleanup()


class SingleCarsDatasetSaver:
    """
    SingleCarsDatasetSaver

    Structure managing the descriptors of each CarsDataset.
    """

    def __init__(self, obj_id, cars_ds):
        """
        Init function of SingleCarsDatasetSaver

        """

        self.obj_id = obj_id
        self.cars_ds = cars_ds

        self.file_names = []
        self.optional_data_list = []
        self.tags = []
        self.dtypes = []
        self.nodatas = []
        self.descriptors = []
        self.save_pc_by_pair_list = []
        self.already_seen = False
        self.count = 0
        self.folder_name = None

    def add_file(
        self,
        file_name,
        tag=None,
        dtype=None,
        nodata=None,
        optional_data=False,
        save_points_cloud_by_pair=False,
    ):
        """
        Add file to current CarsDatasetSaver

        :param file_name: file name to save futures to
        :type file_name: str
        :param tag: tag to save
        :type tag: str
        :param dtype: dtype
        :type dtype: str
        :param nodata: no data value
        :type nodata: float
        :param optional_data: True if the data is optionnal
        :type optional_data: bool
        """

        self.file_names.append(file_name)
        self.tags.append(tag)
        self.dtypes.append(dtype)
        self.nodatas.append(nodata)
        self.optional_data_list.append(optional_data)
        self.save_pc_by_pair_list.append(save_points_cloud_by_pair)

    def save(self, future_result):
        """
        Save future result

        :param future_result: xr.Dataset or pandas.DataFrame

        """

        try:
            if self.cars_ds.dataset_type == "arrays":
                if not self.already_seen:
                    self.add_confidences(future_result)

                    # generate descriptors
                    for count, file_name in enumerate(self.file_names):
                        if self.tags[count] in future_result.keys():
                            desc = self.cars_ds.generate_descriptor(
                                future_result,
                                file_name,
                                tag=self.tags[count],
                                dtype=self.dtypes[count],
                                nodata=self.nodatas[count],
                            )
                            self.descriptors.append(desc)
                        else:
                            self.descriptors.append(None)
                    self.already_seen = True

                for count, file_name in enumerate(self.file_names):
                    if self.tags[count] in future_result.keys():
                        self.cars_ds.run_save(
                            future_result,
                            file_name,
                            tag=self.tags[count],
                            descriptor=self.descriptors[count],
                        )
                    else:
                        log_message = "{} is not consistent.".format(
                            self.tags[count].capitalize()
                        )
                        if self.optional_data_list[count]:
                            logging.debug(log_message)
                        else:
                            logging.warning(log_message)
            elif self.cars_ds.dataset_type == "points":
                # type points
                if not self.already_seen:
                    # get the confidence tags available in future result
                    self.add_confidences(future_result)

                    # create tmp_folder
                    self.folder_name = self.file_names[0]
                    if not os.path.exists(self.folder_name):
                        os.makedirs(self.folder_name)
                    self.already_seen = True

                self.cars_ds.run_save(
                    future_result,
                    os.path.join(self.folder_name, repr(self.count)),
                    overwrite=not self.already_seen,
                    save_points_cloud_by_pair=self.save_pc_by_pair_list[0],
                )
                self.count += 1

            else:
                logging.error(
                    "Saving {} CarsDataset not implemeted".format(
                        self.cars_ds.dataset_type
                    )
                )

        except:  # pylint: disable=W0702 # noqa: B001, E722
            logging.error(traceback.format_exc())
            logging.error("Tile not saved")

    def add_confidences(self, future_result):
        """
        Add all confidence data in the register
        Read confidence from future result outputs and rewrite
        the confidence registered values
        """

        def test_conf(val):
            """
            Check if val key string contains confidence subtring
            """
            if isinstance(val, str):
                return "confidence" in val
            return False

        confidence_tags = list(filter(test_conf, future_result.keys()))
        index = None
        if "confidence" in self.tags:
            # get the confidence indexes in the registered tag
            index_table = [
                idx
                for idx, value in enumerate(self.tags)
                if value == "confidence"
            ]  # self.tags.index("confidence")
            for index in reversed(index_table):
                ref_confidence_path = self.file_names[index]
                confidence_dtype = self.dtypes[index]
                confidence_nodatas = self.nodatas[index]
                # delete the generic confidence registered values
                self.tags.pop(index)
                self.dtypes.pop(index)
                self.nodatas.pop(index)
                self.file_names.pop(index)

                for item in confidence_tags:
                    self.tags.append(item)
                    self.file_names.append(
                        ref_confidence_path.replace(
                            "confidence", item.replace(".", "_")
                        )
                    )
                    self.dtypes.append(confidence_dtype)
                    self.nodatas.append(confidence_nodatas)

    def cleanup(self):
        """
        Cleanup function

        Close properly all opened files

        """

        # close raster files
        for desc in self.descriptors:
            if desc is not None:
                desc.close()

        # TODO merge point clouds ?

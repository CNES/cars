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
Contains functions for wrapper disk
"""

# Standard imports
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from multiprocessing.pool import ThreadPool

import pandas

# Third party imports
import xarray as xr

# CARS imports
from cars.data_structures import cars_dataset, cars_dict
from cars.orchestrator.cluster.mp_cluster.mp_tools import replace_data_rec

# Third party imports


DENSE_NAME = "DenseDO"
SPARSE_NAME = "SparseDO"
DICT_NAME = "DictDO"


class AbstractWrapper(metaclass=ABCMeta):
    """
    AbstractWrapper
    """

    @abstractmethod
    def get_obj(self, obj):
        """
        Get Object

        :param obj: object to transform

        :return: object
        """

    @abstractmethod
    def get_function_and_kwargs(self, func, kwargs, nout=1):
        """
        Get function to apply and overloaded key arguments

        :param func: function to run
        :param kwargs: key arguments of func
        :param nout: number of outputs

        :return: function to apply, overloaded key arguments
        """

    @abstractmethod
    def cleanup(self):
        """
        Cleanup tmp_dir
        """

    @abstractmethod
    def cleanup_future_res(self, future_res):
        """
        Cleanup future result

        :param future_res: future result to clean
        """


class WrapperNone(AbstractWrapper):
    """
    AbstractWrapper
    """

    def __init__(self, tmp_dir):
        """
        Init function of WrapperDisk
        :param tmp_dir: temporary directory
        """

    def get_obj(self, obj):
        """
        Get Object

        :param obj: object to transform

        :return: object
        """
        return obj

    def get_function_and_kwargs(self, func, kwargs, nout=1):
        """
        Get function to apply and overloaded key arguments

        :param func: function to run
        :param kwargs: key arguments of func
        :param nout: number of outputs

        :return: function to apply, overloaded key arguments
        """

        # apply disk wrapper
        new_func = none_wrapper_fun

        # Get overloaded key arguments

        new_kwargs = kwargs
        new_kwargs["fun"] = func

        return new_func, kwargs

    def cleanup(self):
        """
        Cleanup tmp_dir
        """

    def cleanup_future_res(self, future_res):
        """
        Cleanup future result

        :param future_res: future result to clean
        """
        del future_res


class WrapperDisk(AbstractWrapper):

    """
    WrapperDisk
    """

    def __init__(self, tmp_dir):
        """
        Init function of WrapperDisk
        :param tmp_dir: temporary directory
        """
        self.tmp_dir = os.path.join(tmp_dir, "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.current_object_id = 0

        # Create a thead pool for removing data
        self.removing_pool = ThreadPool(1)

    def cleanup(self):
        """
        Cleanup tmp_dir
        """

        logging.info("Clean removing thread pool ...")
        self.removing_pool.close()
        self.removing_pool.join()

        logging.info("Clean tmp directory ...")
        removing_disk_data(self.tmp_dir)

    def cleanup_future_res(self, future_res):
        """
        Cleanup future result

        :param future_res: future result to clean
        """

        if isinstance(future_res, tuple):
            for future_res_i in future_res:
                if is_dumped_object(future_res_i):
                    self.removing_pool.apply_async(
                        removing_disk_data, args=[future_res_i]
                    )

        else:
            if is_dumped_object(future_res):
                self.removing_pool.apply_async(
                    removing_disk_data, args=[future_res]
                )

    def get_function_and_kwargs(self, func, kwargs, nout=1):
        """
        Get function to apply and overloaded key arguments

        :param func: function to run
        :param kwargs: key arguments of func
        :param nout: number of outputs

        :return: function to apply, overloaded key arguments
        """

        # apply disk wrapper
        new_func = disk_wrapper_fun

        # Get overloaded key arguments
        # Create ids
        id_list = []
        for _ in range(nout):
            id_list.append(self.current_object_id)
            self.current_object_id += 1
        new_kwargs = kwargs
        new_kwargs["id_list"] = id_list
        new_kwargs["fun"] = func
        new_kwargs["tmp_dir"] = self.tmp_dir

        return new_func, new_kwargs

    def get_obj(self, obj):
        """
        Get Object

        :param obj: object to transform

        :return: object
        """
        res = load(obj)
        return res


def removing_disk_data(path):
    """
    Remove directory from disk

    :param path: path to delete
    """
    shutil.rmtree(path)


def none_wrapper_fun(*argv, **kwargs):
    """
    Create a wrapper for functionn running it

    :param argv: args of func
    :param kwargs: kwargs of func

    :return: path to results
    """

    func = kwargs["fun"]
    kwargs.pop("fun")
    return func(*argv, **kwargs)


def disk_wrapper_fun(*argv, **kwargs):
    """
    Create a wrapper for function

    :param argv: args of func
    :param kwargs: kwargs of func

    :return: path to results
    """

    # Get function to wrap and id_list
    try:
        id_list = kwargs["id_list"]
        func = kwargs["fun"]
        tmp_dir = kwargs["tmp_dir"]
        kwargs.pop("id_list")
        kwargs.pop("fun")
        kwargs.pop("tmp_dir")
    except Exception as exc:  # pylint: disable=W0702 # noqa: B001, E722
        raise RuntimeError(
            "Failed in unwrapping. \n Args: {}, \n Kwargs: {}\n".format(
                argv, kwargs
            )
        ) from exc

    # load args
    loaded_argv = load_args_or_kwargs(argv)
    loaded_kwargs = load_args_or_kwargs(kwargs)

    # call function
    res = func(*loaded_argv[:], **loaded_kwargs)

    if res is not None:
        to_disk_res = dump(res, tmp_dir, id_list)
    else:
        to_disk_res = res

    return to_disk_res


def load_args_or_kwargs(args_or_kwargs):
    """
    Load args or kwargs from disk to memory

    :param args_or_kwargs: args or kwargs of func

    :return: new args

    """

    def transform_path_to_obj(obj):
        """
        Transform path to object

        :param obj: object

        """
        res = obj
        if is_dumped_object(obj):
            res = load(obj)

        return res

    # replace data
    return replace_data_rec(args_or_kwargs, transform_path_to_obj)


def is_dumped_object(obj):
    """
    Check if a given object is dumped

    :param obj: object

    :return: is dumped
    :rtype: bool
    """

    is_dumped = False
    if isinstance(obj, str):
        if DENSE_NAME in obj or SPARSE_NAME in obj or DICT_NAME in obj:
            is_dumped = True

    return is_dumped


def load(path):
    """
    Load object from disk

    :param path: path
    :type path: str

    :return: object
    """

    if path is not None:
        obj = path
        if DENSE_NAME in path:
            obj = cars_dataset.CarsDataset("arrays").load_single_tile(path)

        elif SPARSE_NAME in path:
            obj = cars_dataset.CarsDataset("points").load_single_tile(path)
        elif DICT_NAME in path:
            obj = cars_dataset.CarsDataset("dict").load_single_tile(path)

        else:
            logging.warning("Not a dumped arrays or points or dict")

    else:
        obj = None
    return obj


def dump_single_object(obj, path):
    """
    Dump object to disk

    :param path: path
    :type path: str
    """

    if isinstance(obj, xr.Dataset):
        # is from array
        cars_dataset.CarsDataset("arrays").save_single_tile(obj, path)
    elif isinstance(obj, pandas.DataFrame):
        # is from points
        cars_dataset.CarsDataset("points").save_single_tile(obj, path)
    elif isinstance(obj, cars_dict.CarsDict):
        # is from points
        cars_dataset.CarsDataset("dict").save_single_tile(obj, path)
    else:
        raise TypeError("Not an arrays or points or dict")


def create_path(obj, tmp_dir, id_num):
    """
    Create path where to dump object

    :param tmp_dir: tmp_dir
    :param id_num: id of object

    :return: path
    """

    path = None

    if isinstance(obj, xr.Dataset):
        # is from array
        path = DENSE_NAME
    elif isinstance(obj, pandas.DataFrame):
        # is from points
        path = SPARSE_NAME
    elif isinstance(obj, cars_dict.CarsDict):
        # is from dict
        path = DICT_NAME
    else:
        logging.warning("Not an arrays or points or dict")
        path = obj

    path = os.path.join(tmp_dir, path + "_" + repr(id_num))

    return path


def dump(res, tmp_dir, id_list):
    """
    Dump results to tmp_dir, according to ids

    :param res: objects to dump
    :param tmp_dir: tmp_dir
    :param id_list: list of ids of objects

    :return: path
    """

    paths = None

    if len(id_list) > 1:
        paths = []
        for i, single_id in enumerate(id_list):
            if res[i] is not None:
                path = create_path(res[i], tmp_dir, single_id)
                dump_single_object(res[i], path)
                paths.append(path)
            else:
                paths.append(None)

        paths = (*paths,)

    else:
        paths = create_path(res, tmp_dir, id_list[0])
        dump_single_object(res, paths)

    return paths

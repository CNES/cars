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
Contains class objects used by multiprocessing cluster
"""

# Standard imports
import threading


class MpJob:  # pylint: disable=R0903
    """
    Encapsulation of multiprocessing job Id (internal use for mp_local_cluster)
    """

    __slots__ = ["task_id", "r_idx"]

    def __init__(self, idx, return_index):
        self.task_id = idx
        self.r_idx = return_index


class MpDelayedTask:  # pylint: disable=R0903
    """
    Delayed task
    """

    def __init__(self, func, args, kw_args):
        """
        Init function of MpDelayedTask

        :param func: function to run
        :param args: args of function
        :param kw_args: kwargs of function

        """
        self.func = func
        self.args = args
        self.kw_args = kw_args
        self.associated_objects = []

    def __repr__(self):
        """
        Repr function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def __str__(self):
        """
        Str function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def custom_print(self):
        """
        Return string of self
        :return : printable delayed
        """

        res = (
            "MpDelayedTask: \n "
            + str(self.func)
            + "\n"
            + "args : "
            + str(self.args)
            + "\n kw_args:  \n"
            + str(self.kw_args)
        )

        return res

    def modify_delayed_task(self, wrapper):
        """
        Modify delayed to add wrapper (disk, None)

        :param wrapper: wrapper function
        :type wrapper: fun
        """
        used_func, used_kwargs = wrapper.get_function_and_kwargs(
            self.func, self.kw_args, nout=len(self.associated_objects)
        )

        self.kw_args = used_kwargs
        self.func = used_func


class MpDelayed:  # pylint: disable=R0903
    """
    multiprocessing version of dask.delayed
    """

    def __init__(self, delayed_task, return_index=0):
        self.delayed_task = delayed_task
        self.return_index = return_index

        # register to delayed_task
        self.delayed_task.associated_objects.append(self)

    def __repr__(self):
        """
        Repr function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def __str__(self):
        """
        Str function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def custom_print(self):
        """
        Return string of self
        :return : printable delayed
        """

        res = (
            ("MpDELAYED : \n " + str(self.delayed_task.func) + "\n")
            + "return index: "
            + str(self.return_index)
            + "\n Associated objects:  \n"
            + str(self.delayed_task.associated_objects)
        )

        return res

    def get_depending_delayed(self):
        """
        Get all the delayed that current delayed depends on

        :return list of depending delayed
        :rtype: list(MpDelayed)
        """

        depending_delayed = []

        for arg in self.delayed_task.args:
            if isinstance(arg, MpDelayed):
                depending_delayed.append(arg)

        for kw_arg in self.delayed_task.kw_args:
            if isinstance(kw_arg, MpDelayed):
                depending_delayed.append(kw_arg)

        return depending_delayed


class MpFuture:
    """
    Multiprocessing version of distributed.future
    """

    def __init__(self, mp_future_task, return_index):
        """
        Init function of SequentialCluster

        :param mp_future_task: Future task
        :param return_index: index of return object

        """

        self.mp_future_task = mp_future_task
        # register itself to future_task
        self.mp_future_task.associated_futures.append(self)

        self.result = None
        self._success = None
        self.return_index = return_index
        self.event = threading.Event()

    def cleanup(self):
        """
        Cleanup future
        """
        self.event.clear()

    def ready(self):
        """
        Check if future is ready

        """
        return self.event.is_set()

    def successful(self):
        """
        Check if future is successful

        """
        if not self.ready():
            raise ValueError("mp_future not ready!")
        return self._success

    def set(self, success, obj):
        """
        Set results to future

        :param success: success of future
        :type success: bool
        :param obj: result

        """
        self._success = success
        if self._success:
            if not isinstance(obj, tuple):
                if self.return_index > 0:
                    raise ValueError("Asked for index > 0 in a singleton")
                self.result = obj
            else:
                self.result = obj[self.return_index]
        else:
            self.result = obj
        self.event.set()

    def wait(self, timeout=None):
        """
        Wait

        :param timeout: timeout to apply

        """
        self.event.wait(timeout)

    def get(self, timeout=None):
        """
        Get result

        :param timeout: timeout to apply

        """
        self.wait(timeout)
        if not self.ready():
            raise TimeoutError
        if not self._success:
            raise self.result
        return self.result


class MpFutureIterator:
    """
    iterator on multiprocessing.pool.AsyncResult, similar to as_completed
    Only returns the actual results, delete the future after usage
    """

    def __init__(self, future_list, cluster):
        """
        Init function of MpFutureIterator

        :param future_list: list of futures

        """
        self.future_list = future_list
        self.cluster = cluster
        self.was_killed = False

        # update future list for cleaning
        for future in future_list:
            self.cluster.cl_future_list.append(future)

    def __iter__(self):
        """
        Iterate

        """
        return self

    def __next__(self):
        """
        Next

        """
        if not self.future_list:
            raise StopIteration
        res = None
        while res is None:
            for item in self.future_list:
                if item.ready():
                    if not item.successful():
                        raise RuntimeError("Failure in tasks")
                    res = item
                    break

        self.future_list.remove(res)
        # transform result (depending on the wrapper)
        transformed_res = self.cluster.wrapper.get_obj(res.get())

        # update future list for cleaning
        self.cluster.cl_future_list.remove(res)

        return transformed_res

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

import threading
import time

from cars.orchestrator.cluster.mp_cluster.mp_tools import replace_data_rec


class MpJob:  # pylint: disable=R0903
    """
    Encapsulation of multiprocessing job Id (internal use for mp_local_cluster)
    """

    __slots__ = ["task_id", "r_idx"]

    def __init__(self, idx, return_index):
        self.__class__.__name__ = "MpJob"
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
        self.__class__.__name__ = "MpDelayedTask"
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
        self.__class__.__name__ = "MpDelayed"
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
        if "log_fun" in self.delayed_task.kw_args:
            name = str(self.delayed_task.kw_args["log_fun"])
        elif isinstance(self.delayed_task.args[0], FactorizedObject):
            # Task is factorized
            name = str(self.delayed_task.args[0])
        else:
            name = str(self.delayed_task.func)

        res = (
            ("MpDELAYED : " + name) + " return index: " + str(self.return_index)
        )

        return res

    def get_depending_delayed(self):
        """
        Get all the delayed that current delayed depends on

        :return list of depending delayed
        :rtype: list(MpDelayed)
        """

        def get_depending_delayed_rec(list_or_dict):
            """
            Get all the delayed that current delayed depends on

            :return list of depending delayed
            :rtype: list(MpDelayed)
            """

            depending_delayed = []

            if isinstance(list_or_dict, (list, tuple)):
                for arg in list_or_dict:
                    depending_delayed += get_depending_delayed_rec(arg)

            elif isinstance(list_or_dict, dict):
                for key in list_or_dict:
                    depending_delayed += get_depending_delayed_rec(
                        list_or_dict[key]
                    )

            elif isinstance(list_or_dict, FactorizedObject):
                depending_delayed += get_depending_delayed_rec(
                    list_or_dict.get_args()
                )
                depending_delayed += get_depending_delayed_rec(
                    list_or_dict.get_kwargs()
                )

            elif isinstance(list_or_dict, MpDelayed):
                depending_delayed.append(list_or_dict)

            return depending_delayed

        depending_delayed_in_args = get_depending_delayed_rec(
            self.delayed_task.args
        )
        depending_delayed_in_kwargs = get_depending_delayed_rec(
            self.delayed_task.kw_args
        )

        return depending_delayed_in_args + depending_delayed_in_kwargs


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
        self.__class__.__name__ = "MpFuture"

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

    def __init__(self, future_list, cluster, timeout=None):
        """
        Init function of MpFutureIterator

        :param future_list: list of futures

        """
        self.future_list = future_list
        self.cluster = cluster
        self.was_killed = False
        self.timeout = timeout
        self.past_time = time.time()

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
            if self.timeout is not None:
                if time.time() - self.past_time > self.timeout:
                    raise TimeoutError("No task completed before timeout")
            for item in self.future_list:
                if item.ready():
                    if not item.successful():
                        raise RuntimeError("Failure in tasks")
                    res = item
                    self.past_time = time.time()
                    break

        self.future_list.remove(res)
        # transform result (depending on the wrapper)
        transformed_res = self.cluster.wrapper.get_obj(res.get())

        # update future list for cleaning
        self.cluster.cl_future_list.remove(res)

        return transformed_res


class PreviousData:  # pylint: disable=R0903
    """
    Object used in FactorisedObject for args already computed
    in factorized function

    """

    def __init__(self, delayed):
        """
        Init function of PreviousData

        :param return_index: position of data in output of task
        """
        self.return_index = delayed.return_index


def transform_mp_delayed_to_previous_data(obj):
    """
    Replace MpDelayed by PreviousData object

    :param data: data to replace if necessary

    """

    new_data = obj
    if isinstance(obj, MpDelayed):
        new_data = PreviousData(obj)
    return new_data


def transform_previous_data_to_results(obj, res):
    """
    Replace PreviousData object by real data

    :param data: data to replace if necessary

    """

    new_data = obj
    if isinstance(obj, PreviousData):
        pos = obj.return_index
        if isinstance(res, tuple):
            new_data = res[pos]
        else:
            if pos != 0:
                raise RuntimeError(
                    "Waiting multiple output but res is not tuple"
                )
            new_data = res
    return new_data


class FactorizedObject:
    """
    Object used as args of function factorised_func
    It contains several tasks that can be run within a single function
    """

    def __init__(self, current_task, previous_task):
        """
        Init function of FactorizedObject

        :param current_task: last task to execute in factorized_func
                             (arg of task can be a factorized object)
        :param previous_task: task to add and run before current task
                              (arg of task can NOT be a factorized object)
        """
        current_task_is_factorized = False
        current_factorized_object = None

        current_fun = current_task.func
        current_args = current_task.args
        current_kwargs = current_task.kw_args

        if isinstance(current_args[0], FactorizedObject):
            current_task_is_factorized = True
            current_factorized_object = current_args[0]
            current_args = current_factorized_object.get_args()
            current_kwargs = current_factorized_object.get_kwargs()

        # Replace MpDelayed with PreviousData that will be computed
        # in run method of FactorizedObject before the call of current_task
        new_args = replace_data_rec(
            current_args, transform_mp_delayed_to_previous_data
        )
        new_kwargs = replace_data_rec(
            current_kwargs, transform_mp_delayed_to_previous_data
        )

        if current_task_is_factorized:
            # List of tasks is initialized with all tasks in current
            # factorized task
            current_factorized_object.set_args(new_args)
            current_factorized_object.set_kwargs(new_kwargs)
            self.tasks = current_factorized_object.tasks
        else:
            # List of tasks is initialized with current task
            self.tasks = [
                {
                    "func": current_fun,
                    "args": new_args,
                    "kwargs": new_kwargs,
                }
            ]

        # Add at the end of the list the first task to be executed
        # (self.tasks is a LIFO queue)
        previous_fun = previous_task.func
        previous_args = previous_task.args
        previous_kwargs = previous_task.kw_args

        self.tasks.append(
            {
                "func": previous_fun,
                "args": previous_args,
                "kwargs": previous_kwargs,
            }
        )

    def __str__(self):
        res = "Factorized Object : "
        for task in reversed(self.tasks):
            res += str(task["kwargs"]["log_fun"]) + ", "
        return res

    def get_args(self):
        """
        Get args of first task to execute
        """
        return self.tasks[-1]["args"]

    def set_args(self, args):
        """
        Set args of first task to execute

        :param args: arguments to set
        """
        self.tasks[-1]["args"] = args

    def get_kwargs(self):
        """
        Get kwargs of first task to execute
        """
        return self.tasks[-1]["kwargs"]

    def set_kwargs(self, kwargs):
        """
        Set kwargs of first task to execute

        :param args: keyword arguments to set
        """
        self.tasks[-1]["kwargs"] = kwargs

    def pop_next_task(self, previous_result=None):
        """
        Run the next task to execute, remove it from the list and
        return the result

        :param previous_result: output of previous task
        """
        task = self.tasks.pop()
        func = task["func"]
        args = task["args"]
        kwargs = task["kwargs"]
        if previous_result is not None:
            # Replace PreviousData objects with output of previous task
            args = replace_data_rec(
                args, transform_previous_data_to_results, previous_result
            )
            kwargs = replace_data_rec(
                kwargs, transform_previous_data_to_results, previous_result
            )
        return func(*args, **kwargs)

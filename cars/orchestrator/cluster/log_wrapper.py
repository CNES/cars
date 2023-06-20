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
Contains functions for wrapper logs
"""

import copy
import cProfile
import gc
import io
import logging
import os
import pstats
import time
import uuid
from abc import ABCMeta, abstractmethod
from importlib import import_module

import psutil


# pylint: disable=too-few-public-methods
class AbstractLogWrapper(metaclass=ABCMeta):
    """
    AbstractLogWrapper
    """

    @abstractmethod
    def __init__(self, func):
        """
        Init the log/profiling wrapper (store function name)

        :param func: function to run
        """

    @abstractmethod
    def func_args_plus(self):
        """
        getter for the args of the future function

        :return: function to apply, overloaded key arguments
        """


class LogWrapper(AbstractLogWrapper):
    """
    LogWrapper

    simple log wrapper to eval the function elapsed time
    """

    def __init__(self, func, loop_testing):
        self.used_function = func
        self.loop_testing = loop_testing

    def func_args_plus(self):
        fun = log_function
        new_kwarg = {
            "fun_log_wrapper": self.used_function,
            "loop_testing": self.loop_testing,
        }

        return fun, new_kwarg


class CProfileWrapper(AbstractLogWrapper):
    """
    CProfileWrapper

    log wrapper to analyze the internal time consuming of the function.
    The wrapper use cprofile API.
    """

    def __init__(self, func):
        self.used_function = func

    def func_args_plus(self):
        fun = time_profiling_function
        new_kwarg = {"fun_log_wrapper": self.used_function}

        return fun, new_kwarg


class MemrayWrapper(AbstractLogWrapper):
    """
    MemrayWrapper

    log wrapper to analyze the internal allocation
    memory consuming of the function.
    The wrapper use cprofile API.
    """

    def __init__(self, func, loop_testing, out_dir):
        self.used_function = func
        self.loop_testing = loop_testing
        self.out_dir = out_dir

    def func_args_plus(self):
        fun = memory_profiling_function
        new_kwarg = {
            "fun_log_wrapper": self.used_function,
            "loop_testing": self.loop_testing,
            "out_dir": self.out_dir,
        }

        return fun, new_kwarg


def log_function(*argv, **kwargs):
    """
    Create a wrapper for function running it

    :param argv: args of func
    :param kwargs: kwargs of func

    :return: path to results
    """
    func = kwargs["fun_log_wrapper"]
    loop_testing = kwargs["loop_testing"]
    kwargs.pop("fun_log_wrapper")
    kwargs.pop("loop_testing")
    start_time = time.time()

    memory_start = get_current_memory()

    if loop_testing:
        res = loop_function(argv, kwargs, func)
    else:
        res = func(*argv, **kwargs)
    total_time = time.time() - start_time
    switch_messages(func, total_time)

    memory_end = get_current_memory()
    log_delta_memory(func, memory_start, memory_end)

    return res


def time_profiling_function(*argv, **kwargs):
    """
    Create a wrapper to profile the function elapse time

    :param argv: args of func
    :param kwargs: kwargs of func

    :return: path to results
    """
    func = kwargs["fun_log_wrapper"]
    kwargs.pop("fun_log_wrapper")
    # Monitor time
    start_time = time.time()
    # Profile time
    profiler = cProfile.Profile()
    profiler.enable()
    res = func(*argv, **kwargs)
    profiler.disable()
    total_time = time.time() - start_time

    switch_messages(func, total_time)
    print("## PROF STATs")

    stream_cumtime = io.StringIO()
    stream_calls = io.StringIO()
    pstats.Stats(profiler, stream=stream_cumtime).sort_stats(
        "tottime"
    ).print_stats(5)
    pstats.Stats(profiler, stream=stream_calls).sort_stats("calls").print_stats(
        5
    )
    logging.info(stream_cumtime.getvalue())
    print(stream_cumtime.getvalue())
    logging.info(stream_calls.getvalue())
    print(stream_calls.getvalue())
    print("----------")
    return res


def memory_profiling_function(*argv, **kwargs):
    """
    Create a wrapper to profile the function occupation memory

    :param argv: args of func
    :param kwargs: kwargs of func

    :return: path to results
    """
    func = kwargs["fun_log_wrapper"]
    loop_testing = kwargs["loop_testing"]
    outputdir = kwargs["out_dir"]

    kwargs.pop("fun_log_wrapper")
    kwargs.pop("loop_testing")
    kwargs.pop("out_dir")

    # Monitor time
    memray = import_module("memray")
    start_time = time.time()
    unique_filename = str(uuid.uuid4())
    # Profile memory
    with memray.Tracker(
        os.path.join(
            outputdir,
            "profiling",
            "memray",
            func.__name__ + "-" + unique_filename + ".bin",
        )
    ):
        if loop_testing:
            res = loop_function(argv, kwargs, func)
        else:
            res = func(*argv, **kwargs)
    total_time = time.time() - start_time

    switch_messages(func, total_time)
    print("----------")
    return res


def switch_messages(func, total_time):
    """
    create profile message with specific message
    depends on elapsed time (LONG, FAST...).


    :param func : profiled function
    :param total_time : elapsed time of the function
    """
    if total_time >= 1:
        message = "# {}: {:.3f} s LONG".format(
            func.__name__.capitalize(), total_time
        )
        log_message(func, message)
    elif 1 > total_time >= 0.001:
        message = "# {}: {:.4f} s FAST".format(
            func.__name__.capitalize(), total_time
        )
        log_message(func, message)
    elif 0.001 > total_time >= 0.000001:
        message = "# {}: {:.4f} ms VERY FAST".format(
            func.__name__.capitalize(), total_time * 1000.0
        )
        log_message(func, message)
    else:
        message = "# {}: TOO FAST".format(func.__name__.capitalize())
        log_message(func, message)


def log_message(func, message):
    """
    log profiling message

    :param func : logged function
    :param message : log message
    """
    logging.info(message)
    logging.info(func.__module__)
    print(message)
    print(func.__module__)


def loop_function(argv, kwargs, func, nb_iteration=5):
    """
    generate a loop on each cluster function to eval possible leak

    :param argv : input argv
    :param kwargs : input kwargs
    :param func : function to evaluation
    :param nb_iteration (int, optional): number of the iteration loop.
    :param Defaults to 5.

    Returns:
        _type_: result of the function
    """
    logging.info("{} {}".format(func.__module__, func.__name__.capitalize()))
    argv_temp = copy.copy(argv)
    kwargs_temp = copy.deepcopy(kwargs)
    # execute sevral time the function to observe possible leaks
    for k in range(1, nb_iteration):
        logging.info("loop iteration {}".format(k))
        func(*argv, **kwargs)
        del argv
        del kwargs
        gc.collect()
        argv = copy.deepcopy(argv_temp)
        kwargs = copy.deepcopy(kwargs_temp)
    return func(*argv, **kwargs)


def get_current_memory():
    """
    Get current memory of process

    :return: memory
    :rtype: float

    """

    # Use psutil to capture python process memory as well
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss

    # Convert nbytes size for logger
    process_memory = float(process_memory) / 1000000

    return process_memory


def log_delta_memory(func, memory_start, memory_end):
    """
    Log memory infos

    :param func: profiled function
    :param memory_start: memory before the run of function
    :type memory_start: float
    :param memory_end: memory after the run of function
    :type memory_end: float

    """

    message = "Memory before run: {}Mb, Memory after run: {}Mb".format(
        str(memory_start), str(memory_end)
    )

    log_message(func, message)

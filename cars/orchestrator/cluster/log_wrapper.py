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
# pylint: disable=too-many-lines

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
from multiprocessing import Pipe
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from json_checker import Checker

from cars.core import cars_logging
from cars.core.utils import safe_makedirs

THREAD_TIMEOUT = 2


# pylint: disable=too-few-public-methods
class AbstractLogWrapper(metaclass=ABCMeta):
    """
    AbstractLogWrapper
    """

    available_modes = {}

    def __new__(cls, conf_profiling, out_dir):  # pylint: disable=W0613
        """
        Return Log wrapper
        """
        profiling_mode = "cars_profiling"

        if "mode" not in conf_profiling:
            logging.debug("Profiling mode not defined, default is used")
        else:
            profiling_mode = conf_profiling["mode"]

        return super(AbstractLogWrapper, cls).__new__(
            cls.available_modes[profiling_mode]
        )

    def __init__(self, conf_profiling, out_dir):
        # Check conf
        self.checked_conf_profiling = self.check_conf(conf_profiling)
        self.out_dir = out_dir

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name
        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods
            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.available_modes[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

    @abstractmethod
    def get_func_args_plus(self, func):
        """
        getter for the args of the future function

        :param: function to apply

        :return: function to apply, overloaded key arguments
        """


@AbstractLogWrapper.register_subclass("cars_profiling")
class LogWrapper(AbstractLogWrapper):
    """
    LogWrapper

    simple log wrapper doing nothing
    """

    def __init__(self, conf_profiling, out_dir):
        self.out_dir = out_dir
        # call parent init
        super().__init__(conf_profiling, out_dir)
        self.loop_testing = self.checked_conf_profiling["loop_testing"]

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
        overloaded_conf["mode"] = conf.get("mode", "cars_profiling")
        overloaded_conf["loop_testing"] = conf.get("loop_testing", False)

        cluster_schema = {"mode": str, "loop_testing": bool}

        # Check conf
        checker = Checker(cluster_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_func_args_plus(self, func):
        fun = log_function
        new_kwarg = {
            "fun_log_wrapper": func,
            "loop_testing": self.loop_testing,
        }

        return fun, new_kwarg


@AbstractLogWrapper.register_subclass("cprofile")
class CProfileWrapper(AbstractLogWrapper):
    """
    CProfileWrapper

    log wrapper to analyze the internal time consuming of the function.
    The wrapper use cprofile API.
    """

    def __init__(self, conf_profiling, out_dir):
        self.out_dir = out_dir
        # call parent init
        super().__init__(conf_profiling, out_dir)

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
        overloaded_conf["mode"] = conf.get("mode", "cars_profiling")
        cluster_schema = {"mode": str}

        # Check conf
        checker = Checker(cluster_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_func_args_plus(self, func):
        fun = time_profiling_function
        new_kwarg = {"fun_log_wrapper": func}

        return fun, new_kwarg


@AbstractLogWrapper.register_subclass("memray")
class MemrayWrapper(AbstractLogWrapper):
    """
    MemrayWrapper

    log wrapper to analyze the internal allocation
    memory consuming of the function.
    The wrapper use cprofile API.
    """

    def __init__(self, conf_profiling, out_dir):
        self.out_dir = out_dir
        profiling_memory_dir = os.path.join(out_dir, "profiling", "memray")
        safe_makedirs(profiling_memory_dir, cleanup=True)
        # call parent init
        super().__init__(conf_profiling, out_dir)
        self.loop_testing = self.checked_conf_profiling["loop_testing"]

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
        overloaded_conf["mode"] = conf.get("mode", "cars_profiling")
        overloaded_conf["loop_testing"] = conf.get("loop_testing", False)

        cluster_schema = {"mode": str, "loop_testing": bool}

        # Check conf
        checker = Checker(cluster_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_func_args_plus(self, func):
        fun = memory_profiling_function
        new_kwarg = {
            "fun_log_wrapper": func,
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

    if loop_testing:
        # Profile
        res = cars_profile(name=func.__name__ + "_looped", interval=0.2)(
            loop_function
        )(argv, kwargs, func)
    else:
        res = cars_profile(interval=0.2)(func)(*argv, **kwargs)

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


def switch_messages(func, total_time, max_memory=None):
    """
    create profile message with specific message
    depends on elapsed time (LONG, FAST...).


    :param func : profiled function
    :param total_time : elapsed time of the function
    """
    message = "Clock# %{}%: %{:.4f}% s Max ram : {} MiB".format(
        func.__name__.capitalize(), total_time, max_memory
    )

    if total_time >= 1:
        message += " LONG"
    elif 1 > total_time >= 0.001:
        message += " FAST"
    elif 0.001 > total_time >= 0.000001:
        message += " VERY FAST"
    else:
        message += " TOO FAST"

    log_message(func, message)


def log_message(func, message):
    """
    log profiling message

    :param func : logged function
    :param message : log message
    """
    cars_logging.add_profiling_message(message)
    cars_logging.add_profiling_message(func.__module__)


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


def generate_summary(out_dir, used_conf):
    """
    Generate Profiling summary
    """

    nb_workers = 1
    if "nb_workers" in used_conf["orchestrator"]:
        nb_workers = used_conf["orchestrator"]["nb_workers"]

    log_file_main = os.path.join(
        out_dir,
        "workers_log",
        "profiling.log",
    )

    out_profiling_main = os.path.join(out_dir, "profiling", "profiling.log")

    log_files = [log_file_main, out_profiling_main]

    names = []
    times = []
    max_ram = []
    start_ram = []
    end_ram = []
    max_cpu = []

    for log_file in log_files:
        if not os.path.exists(log_file):
            logging.debug("{} log file does not exist".format(log_file))
            return

        with open(log_file, encoding="UTF-8") as file_desc:
            for item in file_desc:
                if "CarsProfiling" in item:
                    splited_items = item.split("%")
                    names.append(splited_items[1])
                    times.append(float(splited_items[3]))
                    max_ram.append(float(splited_items[5]))
                    start_ram.append(float(splited_items[7]))
                    end_ram.append(float(splited_items[9]))
                    max_cpu.append(float(splited_items[11]))

    times_df = pd.DataFrame(
        {
            "name": names,
            "time": times,
            "max_ram": max_ram,
            "start_ram": start_ram,
            "end_ram": end_ram,
            "max_cpu": max_cpu,
        }
    )

    # Generate summary message
    cars_logging.add_profiling_message(
        "\n \n \n "
        "----------------------------------------"
        " SUMMARY PROFILLING  "
        "----------------------------------------"
        " \n \n \n"
    )

    summary_names = []
    summary_max_ram = []
    summary_max_ram_err_min = []
    summary_max_ram_err_max = []
    summary_max_ram_relative = []
    summary_mean_time_per_task = []
    summary_mean_time_per_task_err_min = []
    summary_mean_time_per_task_err_max = []
    summary_total_time = []
    summary_max_cpu = []
    summary_nb_calls = []
    full_max_ram = []
    full_added_ram = []
    full_time = []
    full_max_cpu = []

    for name in pd.unique(times_df["name"]):
        current_df = times_df.loc[times_df["name"] == name]

        current_med_time = current_df["time"].mean()
        current_med_time_err_min = current_med_time - current_df["time"].min()
        current_med_time_err_max = current_df["time"].max() - current_med_time
        total_time = current_df["time"].sum()
        max_ram = current_df["max_ram"].mean()
        max_ram_err_min = max_ram - current_df["max_ram"].min()
        max_ram_err_max = current_df["max_ram"].max() - max_ram
        max_cpu = current_df["max_cpu"].max()
        max_ram_without_start = (
            current_df["max_ram"] - current_df["start_ram"]
        ).max()
        diff_end_start = (current_df["end_ram"] - current_df["start_ram"]).max()
        nb_values = len(current_df)

        # Fill lists with all data
        full_max_ram.append(list(current_df["max_ram"]))
        full_added_ram.append(
            list(current_df["max_ram"] - current_df["start_ram"])
        )
        full_time.append(list(current_df["time"]))
        full_max_cpu.append(list(current_df["max_cpu"]))

        # fill lists for figures
        summary_names.append(name)
        summary_max_ram.append(max_ram)
        summary_max_ram_err_min.append(max_ram_err_min)
        summary_max_ram_err_max.append(max_ram_err_max)
        summary_max_ram_relative.append(max_ram_without_start)
        summary_mean_time_per_task.append(current_med_time)
        summary_mean_time_per_task_err_min.append(current_med_time_err_min)
        summary_mean_time_per_task_err_max.append(current_med_time_err_max)
        summary_total_time.append(total_time)
        summary_max_cpu.append(max_cpu)
        summary_nb_calls.append(nb_values)

        message = (
            "Task {} ran {} times, with mean time {} sec, "
            "total time: {} sec, Max cpu: {} %"
            " max ram in process during task: {} MiB, "
            "max ram - start ram: {},  "
            " end - start ram : {}".format(
                name,
                nb_values,
                current_med_time,
                total_time,
                max_cpu,
                max_ram,
                max_ram_without_start,
                diff_end_start,
            )
        )

        cars_logging.add_profiling_message(message)

    # Generate png
    _, axs = plt.subplots(4, 2, figsize=(15, 15), layout="tight")
    # Fill

    generate_boxplot(
        axs.flat[0], summary_names, full_max_cpu, "Max CPU usage", "%"
    )
    generate_histo(
        axs.flat[1], summary_names, summary_total_time, "Total Time", "s"
    )

    (
        summary_names_without_pipeline,
        total_full_time_without_pipeline,
    ) = filter_lists(
        summary_names,
        full_time,
        lambda name: "pipeline" not in name,
    )
    generate_boxplot(
        axs.flat[2],
        summary_names_without_pipeline,
        total_full_time_without_pipeline,
        "Time per task",
        "s",
    )

    generate_boxplot(
        axs.flat[3],
        summary_names,
        full_max_ram,
        "Max RAM used",
        "MiB",
    )

    generate_boxplot(
        axs.flat[4],
        summary_names,
        full_added_ram,
        "Max RAM added",
        "MiB",
    )
    generate_histo(
        axs.flat[5],
        summary_names,
        summary_nb_calls,
        "NB calls",
        "calls",
    )

    # Pie chart

    (name_task_workers, summary_workers) = filter_lists(
        summary_names, summary_total_time, lambda name: "wrapper" in name
    )

    (name_task_main, summary_main) = filter_lists(
        summary_names,
        summary_total_time,
        lambda name: "wrapper" not in name and "pipeline" not in name,
    )

    (_, [pipeline_time]) = filter_lists(
        summary_names, summary_total_time, lambda name: "pipeline" in name
    )

    total_time_workers = nb_workers * pipeline_time
    generate_pie_chart(
        axs.flat[6],
        name_task_workers,
        100 * np.array(summary_workers) / total_time_workers,
        "Total time in workers ({} workers) lives "
        "(not always with tasks)".format(nb_workers),
    )

    generate_pie_chart(
        axs.flat[7],
        name_task_main,
        100 * np.array(summary_main) / pipeline_time,
        "Total time in main (with waiting for workers)",
    )

    # file_name
    profiling_plot = os.path.join(
        out_dir,
        "profiling",
        "profiling_plots.pdf",
    )
    plt.savefig(profiling_plot)


def filter_lists(names, data, cond):
    """
    Filter lists with condition on name
    """

    filtered_names = []
    filtered_data = []

    for name, dat in zip(names, data):  # noqa: B905
        if cond(name):
            filtered_names.append(name)
            filtered_data.append(dat)

    return filtered_names, filtered_data


def generate_boxplot(axis, names, data_full, title, data_type):
    """
    Generate boxplot
    """

    axis.boxplot(data_full, vert=False, showfliers=False, labels=names)
    axis.invert_yaxis()
    axis.set_xlabel(data_type)
    axis.set_title(title)


def generate_histo(
    axis, names, data, title, data_type, data_min_err=None, data_max_err=None
):
    """
    Generate histogram
    """
    y_pos = np.arange(len(names))
    if None not in (data_min_err, data_max_err):
        data_min_err = np.array(data_min_err)
        data_max_err = np.array(data_max_err)
        xerr = np.empty((2, data_min_err.shape[0]))
        xerr[0, :] = data_min_err
        xerr[1, :] = data_max_err
        axis.barh(y_pos, data, xerr=xerr, align="center")
    else:
        axis.barh(y_pos, data, align="center")
    axis.set_yticks(y_pos, labels=names)
    axis.invert_yaxis()
    axis.set_xlabel(data_type)
    axis.set_title(title)


def generate_pie_chart(axis, names, data, title):
    """
    Generate pie chart, data in %
    """
    names = list(names)
    data = list(data)

    if np.sum(data) > 100:
        cars_logging.add_profiling_message(
            "Chart: sum of data {}> 100%".format(title)
        )
        title += " (with sum > 100%) "
    else:
        others = 100 - np.sum(data)
        data.append(others)
        names.append("other")

    axis.pie(data, labels=names, autopct="%1.1f%%")
    axis.set_title(title)


def cars_profile(name=None, interval=0.1):
    """
    CARS profiling decorator

    :param: func: function to monitor

    """

    def decorator_generator(func):
        """
        Inner function
        """

        def wrapper_cars_profile(*args, **kwargs):
            """
            Profiling wrapper

            Generate profiling logs of functio, run

            :return: func(*args, **kwargs)

            """
            start_time = time.time()

            memory_start = get_current_memory()

            # Launch memory profiling thread
            child_pipe, parent_pipe = Pipe()
            thread_monitoring = CarsMemProf(
                os.getpid(), child_pipe, interval=interval
            )
            thread_monitoring.start()
            if parent_pipe.poll(THREAD_TIMEOUT):
                parent_pipe.recv()

            res = func(*args, **kwargs)
            total_time = time.time() - start_time

            # end memprofiling monitoring
            parent_pipe.send(0)
            max_memory = None
            max_cpu = None
            if parent_pipe.poll(THREAD_TIMEOUT):
                max_memory = parent_pipe.recv()
            if parent_pipe.poll(THREAD_TIMEOUT):
                max_cpu = parent_pipe.recv()
            memory_end = get_current_memory()

            func_name = name
            if name is None:
                func_name = func.__name__.capitalize()

            message = (
                "CarsProfiling# %{}%: %{:.4f}% s Max ram : %{}% MiB"
                " Start Ram: %{}% MiB, End Ram: %{}% MiB, "
                " Max CPU usage: %{}%".format(
                    func_name,
                    total_time,
                    max_memory,
                    memory_start,
                    memory_end,
                    max_cpu,
                )
            )

            cars_logging.add_profiling_message(message)

            return res

        return wrapper_cars_profile

    return decorator_generator


class CarsMemProf(Thread):
    """
    CarsMemProf

    Profiling thread
    """

    def __init__(self, pid, pipe, interval=0.1):
        """
        Init function of CarsMemProf
        """
        super().__init__()
        self.pipe = pipe
        self.interval = interval
        self.cpu_interval = 0.1
        self.process = psutil.Process(pid)

    def run(self):
        """
        Run
        """

        try:
            max_mem = 0
            max_cpu = 0

            # tell parent profiling is ready
            self.pipe.send(0)
            stop = False
            while True:
                # Get memory
                current_mem = self.process.memory_info().rss

                if current_mem > max_mem:
                    max_mem = current_mem

                # Get cpu max
                current_cpu = self.process.cpu_percent(
                    interval=self.cpu_interval
                )
                if current_cpu > max_cpu:
                    max_cpu = current_cpu

                if stop:
                    break
                stop = self.pipe.poll(self.interval)

            # Convert nbytes size for logger
            self.pipe.send(float(max_mem) / 1000000)
            self.pipe.send(max_cpu)

        except BrokenPipeError:
            logging.debug("broken pipe error in log wrapper ")

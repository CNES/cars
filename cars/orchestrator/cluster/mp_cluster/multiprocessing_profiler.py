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
Contains multiprocessing_profiler class
"""

import logging
import os
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psutil

# Agg backend for non interactive
matplotlib.use("Agg")


RAM_PER_WORKER_CHECK_SLEEP_TIME = 2
INTERVAL_CPU = 0.2
SAVE_TIME = 120

TIME = "time"
MAIN_MEMORY = "main_memory"
MAX_PROCESS_MEMORY = "max_process_memory"
MAIN_AND_PROCESS = "main_and_processes"
TOTAL_PROCESS_MEMORY = "total_process_memory"
AVAILABLE_RAM = "available_ram"
TOTAL_RAM = "total_ram"

MAIN_CPU_USAGE = "main_cpu_usage"
TOTAL_PROCESS_CPU_USAGE = "total Proces_cpu_usage"
MAIN_AND_PROCESS_CPU = "main_and_process_cpu"


class Timer:  # pylint: disable=too-few-public-methods
    """
    Start time
    """

    def __init__(self):
        """
        Init
        """
        self.timer = time.time()


class MultiprocessingProfiler:  # pylint: disable=too-few-public-methods
    """
    MultiprocessingProfiler

    Used to profile memory in processes
    """

    def __init__(
        self, pool, out_dir, max_ram_per_worker, mp_dataframe=None, timer=None
    ):
        """
        Init function of MultiprocessingProfiler

        :param pool: pool process to monitor
        :param out_dir: out_dir to save graph
        :param max_ram_per_worker: max ram per worker to use
        """

        self.main_pid = os.getpid()
        self.pool = pool
        self.out_dir = out_dir
        self.file_plot = os.path.join(
            self.out_dir, "logs", "profiling", "memory_profiling.png"
        )
        self.max_ram_per_worker = max_ram_per_worker

        if mp_dataframe is not None and timer is not None:
            self.memory_data = mp_dataframe
            self.timer = timer
        else:
            self.timer = Timer()
            self.memory_data = pd.DataFrame(
                columns=[
                    TIME,
                    MAIN_MEMORY,
                    MAX_PROCESS_MEMORY,
                    MAIN_AND_PROCESS,
                    TOTAL_PROCESS_MEMORY,
                    AVAILABLE_RAM,
                    TOTAL_RAM,
                    MAIN_CPU_USAGE,
                    TOTAL_PROCESS_CPU_USAGE,
                    MAIN_AND_PROCESS_CPU,
                ]
            )

        # Memory usage of Pool
        self.monitor_thread = threading.Thread(
            target=check_pool_memory_usage,
            args=(
                self.main_pid,
                self.pool,
                self.max_ram_per_worker,
                self.memory_data,
                self.timer,
            ),
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.saver_thread = threading.Thread(
            target=save_figure_in_thread,
            args=(self.memory_data, self.file_plot),
        )
        self.saver_thread.daemon = True
        self.saver_thread.start()

    def save_plot(self):
        """
        Save plots
        """
        logging.info("Save profing plots ...")
        save_data(self.memory_data, self.file_plot)


def get_process_memory(process):
    """
    Get process current memory

    :param process

    :return: memory Mb
    """
    return process.memory_info().rss / (1024 * 1024)


def get_cpu_usage(process):
    """
    Get cpu usage

    :param process: Process to monitor
    """

    try:
        cpu_usage = process.cpu_percent(interval=0.1)
    except Exception:
        cpu_usage = 0

    return cpu_usage


def save_figure_in_thread(to_fill_dataframe, file_path):
    """
    Save data during compute

    :param to_fill_dataframe: dataframe to fill
    :param file_path: path to save path
    """

    while True:
        time.sleep(SAVE_TIME)
        # Save file
        save_data(to_fill_dataframe, file_path)


def check_pool_memory_usage(
    main_process_id, pool, max_ram_per_worker, to_fill_dataframe, timer
):
    """
    Check memory usage of each worker in pool

    :param main_process_id: main process id
    :param pool: pool of worker
    :param max_ram_per_worker: max ram to use per worker
    :param to_fill_dataframe: dataframe to fill
    :param timer: timer
    """
    main_process = psutil.Process(main_process_id)
    start_time = timer.timer

    while True:
        # Get time
        current_time = time.time() - start_time
        minutes = current_time / 60

        # Get available memory
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024 * 1024)
        available = ram.available / (1024 * 1024)

        # Check main process
        main_current_memory = get_process_memory(main_process)
        main_process_cpu = get_cpu_usage(main_process)

        # Check workers
        main_and_processes_total = main_current_memory
        total_memory = 0
        max_process_ram = 0
        processes_cpu = 0
        total_cpu = main_process_cpu

        size_pool = len(pool._pool)  # pylint: disable=protected-access
        sleep_time = max(
            0, RAM_PER_WORKER_CHECK_SLEEP_TIME - INTERVAL_CPU * size_pool
        )
        for worker in pool._pool:  # pylint: disable=protected-access
            pid = worker.pid
            try:
                process = psutil.Process(pid)
                memory_usage_mb = get_process_memory(process)

                # Add to metrics
                max_process_ram = max(max_process_ram, memory_usage_mb)
                total_memory += memory_usage_mb
                processes_cpu += get_cpu_usage(process)

                # Check memory to inform user
                if memory_usage_mb > max_ram_per_worker:
                    logging.info(
                        "Process {} is using {} Mb > "
                        "max_ram_per_worker = {} Mb".format(
                            pid, memory_usage_mb, max_ram_per_worker
                        )
                    )
            except psutil.NoSuchProcess:
                # Process no longer exists
                pass

        main_and_processes_total += total_memory
        available_ram_mb = main_and_processes_total + available

        total_cpu += processes_cpu

        # Add to dataframe
        to_fill_dataframe.loc[len(to_fill_dataframe)] = [
            minutes,
            main_current_memory,
            max_process_ram,
            main_and_processes_total,
            total_memory,
            available_ram_mb,
            total_ram,
            main_process_cpu,
            processes_cpu,
            total_cpu,
        ]

        time.sleep(sleep_time)


def save_data(dataframe, file_path):
    """
    Save dataframe to disk

    :param dataframe: file
    :param file_path:
    """

    fig, axs = plt.subplots(5, 1, figsize=(10, 25))

    axs[0].set_title("Total memory used by CARS  Mb")
    axs[0].set_xlabel("Time (min)")
    axs[0].set_ylabel("Memory (MB)")
    dataframe.plot(
        x=TIME,
        y=MAIN_AND_PROCESS,
        ax=axs[0],
        label="Main + Processes memory",
        color="blue",
    )
    dataframe.plot(
        x=TIME, y=TOTAL_RAM, ax=axs[0], label="Machine max memory", color="red"
    )
    dataframe.plot(
        x=TIME,
        y=AVAILABLE_RAM,
        ax=axs[0],
        label=" total CARS + still Available memory",
        color="green",
    )

    axs[1].set_title("Main CARS Process Memory Mb")
    axs[1].set_xlabel("Time (min)")
    axs[1].set_ylabel("Memory (MB)")
    dataframe.plot(
        x=TIME,
        y=MAIN_MEMORY,
        ax=axs[1],
        label="CARS main process",
        color="blue",
    )
    dataframe.plot(
        x=TIME,
        y=MAX_PROCESS_MEMORY,
        ax=axs[1],
        label="CARS max of workers ",
        color="red",
    )

    axs[2].set_title("CARS workers  Process Memory Mb")
    axs[2].set_xlabel("Time (min)")
    axs[2].set_ylabel("Memory (MB)")
    dataframe.plot(
        x=TIME,
        y=TOTAL_PROCESS_MEMORY,
        ax=axs[2],
        label="Total Process Memory",
        color="blue",
    )

    axs[3].set_title("CARS CPU Usage")
    axs[3].set_xlabel("Time (min)")
    axs[3].set_ylabel("CPU (%")
    dataframe.plot(
        x=TIME,
        y=TOTAL_PROCESS_CPU_USAGE,
        ax=axs[3],
        label="Total Process CPU ",
        color="blue",
    )
    dataframe.plot(
        x=TIME,
        y=MAIN_AND_PROCESS_CPU,
        ax=axs[3],
        label="MAIN + Total Process CPU",
        color="red",
    )

    axs[4].set_title("CARS CPU Usage of main process")
    axs[4].set_xlabel("Time (min)")
    axs[4].set_ylabel("CPU (%")
    dataframe.plot(
        x=TIME,
        y=MAIN_CPU_USAGE,
        ax=axs[4],
        label="Main Process CPU ",
        color="blue",
    )

    plt.savefig(file_path, format="png")

    plt.close(fig)

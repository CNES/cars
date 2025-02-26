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
SAVE_TIME = 120

TIME = "time"
MAIN_MEMORY = "main_memory"
MAX_PROCESS_MEMORY = "max_process_memory"
MAIN_AND_PROCESS = "main_and_processes"
TOTAL_PROCESS_MEMORY = "total_process_memory"
AVAILABLE_RAM = "available_ram"
TOTAL_RAM = "total_ram"


class MultiprocessingProfiler:  # pylint: disable=too-few-public-methods
    """
    MultiprocessingProfiler

    Used to profile memory in processes
    """

    def __init__(self, pool, out_dir, max_ram_per_worker):
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

        self.memory_data = pd.DataFrame(
            columns=[
                TIME,
                MAIN_MEMORY,
                MAX_PROCESS_MEMORY,
                MAIN_AND_PROCESS,
                TOTAL_PROCESS_MEMORY,
                AVAILABLE_RAM,
                TOTAL_RAM,
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


def save_figure_in_thread(to_fill_dataframe, file_path):
    """
    Save data during compute

    :param to_fill_dataframe: dataframe to fill
    :param file_path: path to save path
    """
    time.sleep(20)

    while True:
        # Save file
        save_data(to_fill_dataframe, file_path)
        time.sleep(SAVE_TIME)


def check_pool_memory_usage(
    main_process_id, pool, max_ram_per_worker, to_fill_dataframe
):
    """
    Check memory usage of each worker in pool

    :param main_process_id: main process id
    :param pool: pool of worker
    :param max_ram_per_worker: max ram to use per worker
    :param to_fill_dataframe: dataframe to fill
    """
    main_process = psutil.Process(main_process_id)
    start_time = time.time()

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

        # Check workers
        main_and_processes_total = main_current_memory
        total_memory = 0
        max_process_ram = 0
        for worker in pool._pool:  # pylint: disable=protected-access
            pid = worker.pid
            try:
                process = psutil.Process(pid)
                memory_usage_mb = get_process_memory(process)

                # Add to metrics
                max_process_ram = max(max_process_ram, memory_usage_mb)
                total_memory += memory_usage_mb

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

        # Add to dataframe
        to_fill_dataframe.loc[len(to_fill_dataframe)] = [
            minutes,
            main_current_memory,
            max_process_ram,
            main_and_processes_total,
            total_memory,
            available_ram_mb,
            total_ram,
        ]

        time.sleep(RAM_PER_WORKER_CHECK_SLEEP_TIME)


def save_data(dataframe, file_path):
    """
    Save dataframe to disk

    :param dataframe: file
    :param file_path:
    """

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

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

    plt.savefig(file_path, format="png")

    plt.close(fig)

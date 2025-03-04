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
Contains abstract function for multiprocessing Cluster
"""
# pylint: disable=too-many-lines

import copy
import itertools
import logging
import logging.handlers

# Standard imports
import multiprocessing as mp
import os
import platform
import shutil
import signal
import threading
import time
import traceback
from functools import wraps
from multiprocessing import freeze_support
from queue import Queue

# Third party imports
from json_checker import And, Checker, Or

from cars.core import cars_logging

# CARS imports
from cars.orchestrator.cluster import abstract_cluster
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.orchestrator.cluster.mp_cluster import mp_factorizer, mp_wrapper
from cars.orchestrator.cluster.mp_cluster.mp_objects import (
    FactorizedObject,
    MpDelayed,
    MpDelayedTask,
    MpFuture,
    MpFutureIterator,
    MpJob,
)
from cars.orchestrator.cluster.mp_cluster.mp_tools import replace_data
from cars.orchestrator.cluster.mp_cluster.multiprocessing_profiler import (
    MultiprocessingProfiler,
)

SYS_PLATFORM = platform.system().lower()
IS_WIN = "windows" == SYS_PLATFORM

RUN = 0
TERMINATE = 1

# Refresh time between every iteration, to prevent from freezing
REFRESH_TIME = 0.05

job_counter = itertools.count()


@abstract_cluster.AbstractCluster.register_subclass("mp", "multiprocessing")
class MultiprocessingCluster(abstract_cluster.AbstractCluster):
    """
    MultiprocessingCluster
    """

    # pylint: disable=too-many-instance-attributes
    @cars_profile(name="Multiprocessing orchestrator initialization")
    def __init__(
        self, conf_cluster, out_dir, launch_worker=True, data_to_propagate=None
    ):
        """
        Init function of MultiprocessingCluster

        :param conf_cluster: configuration for cluster

        """

        # TODO: remove message
        if conf_cluster["mode"] == "mp":
            message = (
                " 'mp' keyword has been deprecated, use "
                "'multiprocessing' instead"
            )
            logging.warning(message)

        self.out_dir = out_dir
        # call parent init
        super().__init__(
            conf_cluster,
            out_dir,
            launch_worker=launch_worker,
            data_to_propagate=data_to_propagate,
        )

        # retrieve parameters
        self.nb_workers = self.checked_conf_cluster["nb_workers"]
        self.mp_mode = self.checked_conf_cluster["mp_mode"]
        self.task_timeout = self.checked_conf_cluster["task_timeout"]
        self.max_tasks_per_worker = self.checked_conf_cluster[
            "max_tasks_per_worker"
        ]
        self.dump_to_disk = self.checked_conf_cluster["dump_to_disk"]
        self.per_job_timeout = self.checked_conf_cluster["per_job_timeout"]
        self.profiling = self.checked_conf_cluster["profiling"]
        self.factorize_tasks = self.checked_conf_cluster["factorize_tasks"]
        # Set multiprocessing mode
        self.mp_mode = self.checked_conf_cluster["mp_mode"]

        if IS_WIN:
            self.mp_mode = "spawn"
            logging.warning(
                "{} is not functionnal in windows,"
                "spawn will be used instead".format(self.mp_mode)
            )

        self.launch_worker = launch_worker

        self.tmp_dir = None

        # affinity issues caused by numpy
        if IS_WIN is False:
            os.system(
                "taskset -p 0xffffffff %d  > /dev/null 2>&1" % os.getpid()
            )

        if self.launch_worker:
            # Create wrapper object
            if self.dump_to_disk:
                if self.out_dir is None:
                    raise RuntimeError("Not out_dir provided")
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
                self.tmp_dir = os.path.join(self.out_dir, "tmp_save_disk")
                if not os.path.exists(self.tmp_dir):
                    os.makedirs(self.tmp_dir)
                self.wrapper = mp_wrapper.WrapperDisk(self.tmp_dir)
            else:
                self.wrapper = mp_wrapper.WrapperNone(None)

            # Create pool
            ctx_in_main = mp.get_context(self.mp_mode)
            # import cars for env variables firts
            # import cars pipelines for numba compilation
            ctx_in_main.set_forkserver_preload(["cars", "cars.pipelines"])
            self.pool = ctx_in_main.Pool(
                self.nb_workers,
                initializer=freeze_support,
                maxtasksperchild=self.max_tasks_per_worker,
            )

            self.queue = Queue()
            self.task_cache = {}

            # Variable used for cleaning
            # Clone of iterator future list
            self.cl_future_list = []

            # set the exception hook
            threading.excepthook = log_error_hook

            # Refresh worker
            self.refresh_worker = threading.Thread(
                target=MultiprocessingCluster.refresh_task_cache,
                args=(
                    self.pool,
                    self.task_cache,
                    self.queue,
                    self.per_job_timeout,
                    self.cl_future_list,
                    self.nb_workers,
                    self.wrapper,
                ),
            )
            self.refresh_worker.daemon = True
            self.refresh_worker._state = RUN
            self.refresh_worker.start()

            # Profile pool
            mp_dataframe = None
            timer = None
            if self.data_to_propagate is not None:
                mp_dataframe = self.data_to_propagate.get("mp_dataframe", None)
                timer = self.data_to_propagate.get("mp_timer", None)

            self.profiler = MultiprocessingProfiler(
                self.pool,
                self.out_dir,
                self.checked_conf_cluster["max_ram_per_worker"],
                mp_dataframe=mp_dataframe,
                timer=timer,
            )

            self.data_to_propagate = {
                "mp_dataframe": self.profiler.memory_data,
                "mp_timer": self.profiler.timer,
            }

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

        available_cpu = mp.cpu_count()  # TODO returns full node nb cpus
        # TODO robustify if a partial node is used
        # One process per cpu for memory usage estimated

        # Modify some env variables for memory  usage
        # TODO
        # set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 1

        # Overload conf
        overloaded_conf["mode"] = conf.get("mode", "mp")
        overloaded_conf["mp_mode"] = conf.get("mp_mode", "forkserver")
        nb_workers = conf.get("nb_workers", 2)
        overloaded_conf["nb_workers"] = min(available_cpu, nb_workers)
        overloaded_conf["task_timeout"] = conf.get("task_timeout", 600)
        overloaded_conf["max_ram_per_worker"] = conf.get(
            "max_ram_per_worker", 2000
        )
        overloaded_conf["max_tasks_per_worker"] = conf.get(
            "max_tasks_per_worker", 10
        )
        overloaded_conf["dump_to_disk"] = conf.get("dump_to_disk", True)
        overloaded_conf["per_job_timeout"] = conf.get("per_job_timeout", 600)
        overloaded_conf["factorize_tasks"] = conf.get("factorize_tasks", True)
        overloaded_conf["profiling"] = conf.get("profiling", {})

        cluster_schema = {
            "mode": str,
            "dump_to_disk": bool,
            "mp_mode": str,
            "nb_workers": And(int, lambda x: x > 0),
            "task_timeout": And(int, lambda x: x > 0),
            "max_ram_per_worker": And(Or(float, int), lambda x: x > 0),
            "max_tasks_per_worker": And(int, lambda x: x > 0),
            "per_job_timeout": Or(float, int),
            "profiling": dict,
            "factorize_tasks": bool,
        }

        # Check conf
        checker = Checker(cluster_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_delayed_type(self):
        """
        Get delayed type
        """
        return MpDelayed

    def cleanup(self, keep_shared_dir=False):
        """
        Cleanup cluster
        :param keep_shared_dir: do not clean directory of shared objects
        """

        # Save profiling
        self.profiler.save_plot()

        # clean profiler
        self.profiler.cleanup()

        # Terminate worker
        self.refresh_worker._state = TERMINATE  # pylint: disable=W0212
        while self.refresh_worker.is_alive():
            time.sleep(0)

        # close pool
        self.pool.terminate()
        self.pool.join()

        # clean tmpdir if exists
        self.wrapper.cleanup(keep_shared_dir=keep_shared_dir)

        if not keep_shared_dir:
            if self.tmp_dir is not None:
                shutil.rmtree(self.tmp_dir)

    def scatter(self, data):
        """
        Distribute data through workers

        :param data: data to dump
        """
        return self.wrapper.scatter_obj(data)

    def create_task_wrapped(self, func, nout=1):
        """
        Create task

        :param func: function
        :param nout: number of outputs
        """

        @wraps(func)
        def mp_delayed_builder(*argv, **kwargs):
            """
            Create a MPDelayed builder

            :param argv: args of func
            :param kwargs: kwargs of func
            """
            new_kwargs = kwargs
            new_kwargs["log_dir"] = self.worker_log_dir
            new_kwargs["log_level"] = self.log_level
            new_kwargs["log_fun"] = func
            # create delayed_task
            delayed_task = MpDelayedTask(
                cars_logging.logger_func, list(argv), new_kwargs
            )

            delayed_object_list = []
            for idx in range(nout):
                delayed_object_list.append(
                    MpDelayed(delayed_task, return_index=idx)
                )

            res = None
            if len(delayed_object_list) == 1:
                res = delayed_object_list[0]
            else:
                res = (*delayed_object_list,)

            return res

        return mp_delayed_builder

    def start_tasks(self, task_list):
        """
        Start all tasks

        :param task_list: task list
        """
        memorize = {}
        # Use a copy of input delayed
        task_list = copy.deepcopy(task_list)
        if self.factorize_tasks:
            mp_factorizer.factorize_delayed(task_list)
        future_list = [self.rec_start(task, memorize) for task in task_list]
        # signal that we reached the end of this batch
        self.queue.put("END_BATCH")
        return future_list

    def rec_start(self, delayed_object, memorize):
        """
        Record task

        :param delayed_object: delayed object to record
        :type delayed_object: MpDelayed
        :param memorize: list of MpDelayed already recorded

        """
        # check if this task is already started
        if delayed_object in memorize.keys():
            return memorize[delayed_object]

        can_run = True

        current_delayed_task = delayed_object.delayed_task

        # Modify delayed with wrapper here
        current_delayed_task.modify_delayed_task(self.wrapper)

        def transform_delayed_to_mp_job(args_or_kawargs):
            """
            Replace MpDelayed in list or dict by a MpJob

            :param args_or_kawargs: list or dict of data
            """

            def transform_mp_delayed_to_jobs(obj):
                """
                Replace MpDelayed by MpJob

                :param data: data to replace if necessary
                """

                new_data = obj
                if isinstance(obj, MpDelayed):
                    rec_future = self.rec_start(obj, memorize)
                    new_data = MpJob(
                        rec_future.mp_future_task.job_id,
                        rec_future.return_index,
                    )
                return new_data

            # replace data
            return replace_data(args_or_kawargs, transform_mp_delayed_to_jobs)

        # Transform MpDelayed to MpJob

        filt_args = transform_delayed_to_mp_job(current_delayed_task.args)

        filt_kw = transform_delayed_to_mp_job(current_delayed_task.kw_args)

        # Check if can be run
        dependencies = compute_dependencies(filt_args, filt_kw)
        can_run = True
        if len(dependencies) > 0:
            can_run = False

        # start current task
        task_future = MpFutureTask(self)

        self.queue.put(
            (
                task_future.job_id,
                can_run,
                current_delayed_task.func,
                filt_args,
                filt_kw,
            )
        )

        # Create future object
        object_future = MpFuture(task_future, delayed_object.return_index)
        memorize[delayed_object] = object_future

        # Create other futures associated to this task
        for other_delayed_obj in current_delayed_task.associated_objects:
            if other_delayed_obj != delayed_object:
                memorize[other_delayed_obj] = MpFuture(
                    task_future, other_delayed_obj.return_index
                )

        return object_future

    @staticmethod  # noqa: C901
    def refresh_task_cache(  # noqa: C901
        pool,
        task_cache,
        in_queue,
        per_job_timeout,
        cl_future_list,
        nb_workers,
        wrapper_obj,
    ):
        """
        Refresh task cache

        :param task_cache: task cache list
        :param in_queue: queue
        :param per_job_timeout: per job timeout
        :param cl_future_list: current future list used in iterator
        :param nb_workers:  number of workers
        """
        thread = threading.current_thread()

        # initialize lists
        wait_list = {}
        in_progress_list = {}
        dependencies_list = {}
        done_task_results = {}
        job_ids_to_launch_prioritized = []
        max_nb_tasks_running = 2 * nb_workers

        while thread._state == RUN:  # pylint: disable=W0212
            # wait before next iteration
            time.sleep(REFRESH_TIME)
            # get new task from queue
            if not in_queue.empty():
                # get nb_workers task from this batch
                for job_id, can_run, func, args, kw_args in iter(
                    in_queue.get, "END_BATCH"
                ):
                    wait_list[job_id] = [func, args, kw_args]
                    if can_run:
                        job_ids_to_launch_prioritized.append(job_id)
                        # add to dependencies (-1 to identify initial tasks)
                        dependencies_list[job_id] = [-1]
                    else:
                        # get dependencies
                        dependencies_list[job_id] = compute_dependencies(
                            args, kw_args
                        )
                        if len(dependencies_list[job_id]) == 0:
                            dependencies_list[job_id] = [-1]

            # check for ready results
            done_list = []
            next_priority_tasks = []
            for job_id, job_id_progress in in_progress_list.items():
                if job_id_progress.ready():
                    try:
                        res = job_id_progress.get(timeout=per_job_timeout)
                        success = True
                    except:  # pylint: disable=W0702 # noqa: B001, E722
                        res = traceback.format_exc()
                        success = False
                        logging.error("Exception in worker: {}".format(res))
                    done_list.append(job_id)
                    done_task_results[job_id] = [success, res]

                    # remove from dependance list
                    dependencies_list.pop(job_id)

                    # search related priority task
                    for job_id2 in wait_list.keys():  # pylint: disable=C0201
                        depending_tasks = list(dependencies_list[job_id2])
                        if job_id in depending_tasks:
                            next_priority_tasks += depending_tasks
            # remove duplicate dependance task
            next_priority_tasks = list(dict.fromkeys(next_priority_tasks))
            # clean done jobs
            for job_id in done_list:
                # delete
                del in_progress_list[job_id]
                # copy results to futures
                # (they remove themselves from task_cache
                task_cache[job_id].set(done_task_results[job_id])

            (
                ready_list,
                failed_list,
            ) = MultiprocessingCluster.get_ready_failed_tasks(
                wait_list, dependencies_list, done_task_results
            )

            # add ready task in next_priority_tasks
            priority_list = list(
                filter(lambda job_id: job_id in next_priority_tasks, ready_list)
            )

            job_ids_to_launch_prioritized = update_job_id_priority(
                job_ids_to_launch_prioritized, priority_list, ready_list
            )

            # Deal with failed tasks
            for job_id in failed_list:
                done_list.append(job_id)
                done_task_results[job_id] = [
                    False,
                    "Failed depending task",
                ]
                # copy results to futures
                # (they remove themselves from task_cache
                task_cache[job_id].set(done_task_results[job_id])
                del wait_list[job_id]

            while (
                len(in_progress_list) < max_nb_tasks_running
                and len(job_ids_to_launch_prioritized) > 0
            ):
                job_id = job_ids_to_launch_prioritized.pop()
                func, args, kw_args = wait_list[job_id]
                # replace jobs by real data
                new_args = replace_job_by_data(args, done_task_results)
                new_kw_args = replace_job_by_data(kw_args, done_task_results)
                # launch task
                in_progress_list[job_id] = pool.apply_async(
                    func, args=new_args, kwds=new_kw_args
                )
                del wait_list[job_id]
            # find done jobs that can be cleaned
            cleanable_jobid = []

            for job_id in done_task_results.keys():  # pylint: disable=C0201
                # check if needed
                still_need = False
                for dependance_task_list in dependencies_list.values():
                    if job_id in dependance_task_list:
                        still_need = True
                if not still_need:
                    cleanable_jobid.append(job_id)

            # clean unused in the future jobs through wrapper
            for job_id_to_clean in cleanable_jobid:
                if job_id_to_clean not in get_job_ids_from_futures(
                    cl_future_list
                ):
                    # not needed by iterator -> can be cleaned
                    # Cleanup with wrapper
                    wrapper_obj.cleanup_future_res(
                        done_task_results[job_id_to_clean][1]
                    )
                    # cleanup list
                    done_task_results.pop(job_id_to_clean)

    @staticmethod
    def get_ready_failed_tasks(wait_list, dependencies_list, done_task_results):
        """
        Return the new ready tasks without constraint
        and failed tasks
        """
        ready_list = []
        failed_list = []
        done_task_result_keys = done_task_results.keys()
        for job_id in wait_list.keys():  # pylint: disable=C0201
            depending_tasks = dependencies_list[job_id]
            # check if all tasks are finished
            can_run = True
            failed = False
            for depend in list(filter(lambda dep: dep != -1, depending_tasks)):
                if depend not in done_task_result_keys:
                    can_run = False
                else:
                    if not done_task_results[depend][0]:
                        # not a success
                        can_run = False
                        failed = True
            if failed:
                # Add to done list with failed status
                failed_list.append(job_id)
            if can_run:
                ready_list.append(job_id)
        return ready_list, failed_list

    @staticmethod
    def get_tasks_without_deps(dependencies_list, ready_list, nb_ready_task):
        """
        Return the list of ready tasks without dependencies
        and not considered like initial task (dependance = -1)
        """
        priority_list = []
        for _ in range(nb_ready_task):
            task_id = next(
                filter(
                    lambda job_id: len(dependencies_list[job_id]) != 1
                    and dependencies_list[job_id][0] != -1,
                    ready_list,
                ),
                None,
            )
            if task_id:
                priority_list.append(task_id)
        return priority_list

    def future_iterator(self, future_list, timeout=None):
        """
        Start all tasks

        :param future_list: future_list list
        """

        return MpFutureIterator(future_list, self, timeout=timeout)


def get_job_ids_from_futures(future_list):
    """
    Get list of jobs ids in future list

    :param future_list: list of futures
    :type future_list: MpFuture

    :return: list of job id
    :rtype: list(int)
    """

    list_ids = []

    for future in future_list:
        list_ids.append(future.mp_future_task.job_id)

    return list_ids


def replace_job_by_data(args_or_kawargs, done_task_results):
    """
    Replace MpJob in list or dict by their real data

    :param args_or_kawargs: list or dict of data
    :param done_task_results: dict of done tasks
    """

    def get_data(data, done_task_results):
        """
        Replace MpJob in list or dict by their real data

        :param data: data to replace if necessary
        :param done_task_results: dict of done tasks
        """

        new_data = data
        if isinstance(data, MpJob):
            task_id = data.task_id
            idx = data.r_idx

            full_res = done_task_results[task_id][1]
            if not done_task_results[task_id][0]:
                raise RuntimeError("Current task failed {}".format(full_res))

            if isinstance(full_res, tuple):
                new_data = full_res[idx]
            else:
                if idx > 0:
                    raise ValueError("Asked for index > 0 in a singleton")
                new_data = full_res

        return new_data

    # replace data
    return replace_data(args_or_kawargs, get_data, done_task_results)


def compute_dependencies(args, kw_args):
    """
    Compute dependencies from args and kw_args

    :param args: arguments
    :type args: list
    :param kw_args: key arguments
    :type kw_args: dict

    :return: dependencies
    :rtype: list
    """

    def get_job_id(data):
        """
        Get job id from data if is MpJob

        :param data

        :return job id if exists, None if doesnt exist
        :rtype: int
        """
        job_id = None

        if isinstance(data, MpJob):
            job_id = data.task_id

        return job_id

    def get_ids_rec(list_or_dict):
        """
        Compute dependencies from list or dict or simple data

        :param list_or_dict: arguments
        :type list_or_dict: list or dict

        :return: dependencies
        :rtype: list
        """

        list_ids = []

        if isinstance(list_or_dict, (list, tuple)):
            for arg in list_or_dict:
                list_ids += get_ids_rec(arg)

        elif isinstance(list_or_dict, dict):
            for key in list_or_dict:
                list_ids += get_ids_rec(list_or_dict[key])

        elif isinstance(list_or_dict, FactorizedObject):
            facto_args = list_or_dict.get_args()
            for arg in facto_args:
                list_ids += get_ids_rec(arg)
            facto_kwargs = list_or_dict.get_kwargs()
            for key in facto_kwargs:
                list_ids += get_ids_rec(facto_kwargs[key])

        else:
            current_id = get_job_id(list_or_dict)
            if current_id is not None:
                list_ids.append(current_id)

        return list_ids

    # compute dependencies
    dependencies = get_ids_rec(args) + get_ids_rec(kw_args)

    return list(dict.fromkeys(dependencies))


class MpFutureTask:  # pylint: disable=R0903
    """
    multiprocessing version of distributed.future
    """

    def __init__(self, cluster):
        """
        Init function of MpFutureTask

        :param cluster: mp cluster

        """
        self._cluster = cluster
        self.result = None
        self._success = None
        self.event = threading.Event()
        self.job_id = next(job_counter)

        self.task_cache = cluster.task_cache
        self.task_cache[self.job_id] = self

        self.associated_futures = []

    def set(self, obj):
        """
        Set result to associated delayed object, and clean cache

        :param obj: result object
        :type obj: tuple(bool, Union(dataset, dataframe))

        """
        self._success, self.result = obj

        # set result to all futures
        for future in self.associated_futures:
            future.set(self._success, self.result)

        del self.task_cache[self.job_id]
        self._cluster = None
        self.event.clear()


def log_error_hook(args):
    """
    Exception hook for cluster thread
    """
    exc = "Cluster MP thread failed: {}".format(args.exc_value)
    logging.error(exc)
    # Kill thread
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError(exc)


def update_job_id_priority(
    job_ids_to_launch_prioritized, priority_list, ready_list
):
    """
    Update job to launch list with new priority list and ready list

    :return: updated list
    """

    res = priority_list + ready_list + job_ids_to_launch_prioritized
    res = list(dict.fromkeys(res))

    return res

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

import itertools
import logging
import logging.handlers

# Standard imports
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
import traceback
from multiprocessing import Queue, freeze_support

# Third party imports
from json_checker import And, Checker, Or

from cars.core import cars_logging

# CARS imports
from cars.orchestrator.cluster import abstract_cluster
from cars.orchestrator.cluster.mp_cluster import mp_wrapper
from cars.orchestrator.cluster.mp_cluster.mp_objects import (
    MpDelayed,
    MpDelayedTask,
    MpFuture,
    MpFutureIterator,
    MpJob,
)
from cars.orchestrator.cluster.mp_cluster.mp_tools import replace_data_rec

RUN = 0
TERMINATE = 1

# Refresh time between every iteration, to prevent from freezing
REFRESH_TIME = 0.5

job_counter = itertools.count()


@abstract_cluster.AbstractCluster.register_subclass("mp")
class MultiprocessingCluster(abstract_cluster.AbstractCluster):
    """
    MultiprocessingCluster
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, conf_cluster, out_dir, launch_worker=True):
        """
        Init function of MultiprocessingCluster

        :param conf_cluster: configuration for cluster

        """
        self.out_dir = out_dir
        # call parent init
        super().__init__(conf_cluster, out_dir, launch_worker=launch_worker)

        # retrieve parameters
        self.nb_workers = self.checked_conf_cluster["nb_workers"]
        self.dump_to_disk = self.checked_conf_cluster["dump_to_disk"]
        self.per_job_timeout = self.checked_conf_cluster["per_job_timeout"]
        self.profiling = self.checked_conf_cluster["profiling"]
        # Set multiprocessing mode
        # forkserver is used, to allow OMP to be used in numba
        mp_mode = "forkserver"

        self.launch_worker = launch_worker

        self.tmp_dir = None
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
            self.pool = mp.get_context(mp_mode).Pool(
                self.nb_workers,
                initializer=freeze_support,
                maxtasksperchild=100,
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
                    self.wrapper,
                    self.nb_workers,
                ),
            )
            self.refresh_worker.daemon = True
            self.refresh_worker._state = RUN
            self.refresh_worker.start()

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
        # set max_ram_per_worker = total_ram / nb_worker or 4000 by default
        # set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 1

        # Overload conf
        overloaded_conf["mode"] = conf.get("mode", "mp")
        nb_workers = conf.get("nb_workers", 2)
        overloaded_conf["nb_workers"] = min(available_cpu, nb_workers)
        overloaded_conf["max_ram_per_worker"] = conf.get(
            "max_ram_per_worker", 2000
        )
        overloaded_conf["dump_to_disk"] = conf.get("dump_to_disk", True)
        overloaded_conf["per_job_timeout"] = conf.get("per_job_timeout", 600)

        cluster_schema = {
            "mode": str,
            "dump_to_disk": bool,
            "nb_workers": And(int, lambda x: x > 0),
            "max_ram_per_worker": And(Or(float, int), lambda x: x > 0),
            "per_job_timeout": Or(float, int),
            "profiling": {
                "activated": bool,
                "mode": str,
                "loop_testing": bool,
            },
        }

        # Check conf
        checker = Checker(cluster_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def cleanup(self):
        """
        Cleanup cluster

        """

        # Terminate worker
        self.refresh_worker._state = TERMINATE  # pylint: disable=W0212
        while self.refresh_worker.is_alive():
            time.sleep(0)

        # close pool
        self.pool.close()
        self.pool.join()

        # clean tmpdir if exists
        self.wrapper.cleanup()

        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir)

    def scatter(self, data, broadcast=True):
        """
        Distribute data through workers

        :param data: task data
        """
        return data

    def create_task_wrapped(self, func, nout=1):
        """
        Create task

        :param func: function
        :param nout: number of outputs
        """

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
        delayed_object.delayed_task.modify_delayed_task(self.wrapper)

        def transform_delayed_to_mp_job(args_or_kawargs):
            """
            Replace MpDalayed in list or dict by a MpJob

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
            return replace_data_rec(
                args_or_kawargs, transform_mp_delayed_to_jobs
            )

        # Transform MpDelayed to MpJob

        filt_args = transform_delayed_to_mp_job(current_delayed_task.args)

        filt_kw = transform_delayed_to_mp_job(current_delayed_task.kw_args)

        # Check if can be run
        dependances = compute_dependances(filt_args, filt_kw)
        can_run = True
        if len(dependances) > 0:
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
        wrapper_obj,
        nb_workers,
    ):
        """
        Refresh task cache

        :param task_cache: task cache list
        :param in_queue: queue
        :param per_job_timeout: per job timeout
        :param cl_future_list: current future list used in iterator
        :param wrapper_obj: wrapper (disk or None)
        :param nb_workers:  number of workers
        :type wrapper_obj: AbstractWrapper
        """
        thread = threading.current_thread()

        # initialize lists
        wait_list = {}
        in_progress_list = {}
        dependances_list = {}
        done_task_results = {}
        while thread._state == RUN:  # pylint: disable=W0212
            # wait before next iteration
            time.sleep(REFRESH_TIME)
            # get new task from queue
            if not in_queue.empty():
                # get nb_workers task from this batch
                for job_id, can_run, func, args, kw_args in iter(
                    in_queue.get, "END_BATCH"
                ):
                    if can_run and len(in_progress_list) < nb_workers:
                        in_progress_list[job_id] = pool.apply_async(
                            func, args=args, kwds=kw_args
                        )
                        # add to dependances (-1 to identify initial tasks)
                        dependances_list[job_id] = [-1]
                    else:
                        # add to wait list
                        wait_list[job_id] = [func, args, kw_args]
                        # get dependances
                        dependances_list[job_id] = compute_dependances(
                            args, kw_args
                        )
                        if len(dependances_list[job_id]) == 0:
                            dependances_list[job_id] = [-1]

            # check for ready results
            done_list = []
            next_priority_task = []
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
                    dependances_list.pop(job_id)

                # search related priority task
                for job_id2 in wait_list.keys():  # pylint: disable=C0201
                    depending_tasks = list(dependances_list[job_id2])
                    if job_id in depending_tasks:
                        next_priority_task += depending_tasks
            # remove duplicate dependance task
            next_priority_task = list(dict.fromkeys(next_priority_task))

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
                wait_list, dependances_list, done_task_results
            )

            priority_list = []
            nb_ready_task = nb_workers - len(priority_list)

            priority_list += MultiprocessingCluster.get_tasks_without_deps(
                dependances_list, ready_list, nb_ready_task
            )
            # add ready task in next_priority_task
            priority_list += list(
                filter(lambda job_id: job_id in next_priority_task, ready_list)
            )
            # if the priority task have finished
            # continue with the rest of task (initial task)
            if len(priority_list) == 0:
                priority_list += ready_list
            priority_list = list(dict.fromkeys(priority_list))

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

            # launch tasks ready to run
            for job_id in priority_list:
                if len(in_progress_list) < nb_workers:
                    func, args, kw_args = wait_list[job_id]
                    # replace jobs by real data
                    new_args = replace_job_by_data(args, done_task_results)
                    new_kw_args = replace_job_by_data(
                        kw_args, done_task_results
                    )
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
                for dependance_task_list in dependances_list.values():
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
    def get_ready_failed_tasks(wait_list, dependances_list, done_task_results):
        """
        Return the new ready tasks without constraint
        and failed tasks
        """
        ready_list = []
        failed_list = []
        done_task_result_keys = done_task_results.keys()
        for job_id in wait_list.keys():  # pylint: disable=C0201
            depending_tasks = dependances_list[job_id]
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
    def get_tasks_without_deps(dependances_list, ready_list, nb_ready_task):
        """
        Return the list of ready tasks without dependances
        and not considered like initial task (dependance = -1)
        """
        priority_list = []
        for _ in range(nb_ready_task):
            task_id = next(
                filter(
                    lambda job_id: len(dependances_list[job_id]) != 1
                    and dependances_list[job_id][0] != -1,
                    ready_list,
                ),
                None,
            )
            if task_id:
                priority_list.append(task_id)
        return priority_list

    def future_iterator(self, future_list):
        """
        Start all tasks

        :param future_list: future_list list
        """

        return MpFutureIterator(future_list, self)


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
    return replace_data_rec(args_or_kawargs, get_data, done_task_results)


def compute_dependances(args, kw_args):
    """
    Compute dependances from args and kw_args

    :param args: arguments
    :type args: list
    :param kw_args: key arguments
    :type kw_args: dict

    :return: dependances
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
        Compute dependances from list or dict or simple data

        :param list_or_dict: arguments
        :type list_or_dict: list or dict

        :return: dependances
        :rtype: list
        """

        list_ids = []

        if isinstance(list_or_dict, (list, tuple)):
            for arg in list_or_dict:
                list_ids += get_ids_rec(arg)

        elif isinstance(list_or_dict, dict):
            for key in list_or_dict:
                list_ids += get_ids_rec(list_or_dict[key])

        else:
            current_id = get_job_id(list_or_dict)
            if current_id is not None:
                list_ids.append(current_id)

        return list_ids

    # compute dependances
    dependances = get_ids_rec(args) + get_ids_rec(kw_args)

    return list(dict.fromkeys(dependances))


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
    Exception hool for cluster thread
    """
    exc = "Cluster MP thread failed: {}".format(args.exc_value)
    logging.error(exc)
    # Kill thread
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError(exc)

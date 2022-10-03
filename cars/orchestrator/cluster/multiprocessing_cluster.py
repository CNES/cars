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

# Standard imports
import multiprocessing as mp
import os
import shutil
import threading
import time
from multiprocessing import Queue, freeze_support

from json_checker import Checker, Or

# CARS imports
from cars.orchestrator.cluster import abstract_cluster, wrapper

# Third party imports


RUN = 0
TERMINATE = 1

REFRESH_TIME = 1

job_counter = itertools.count()


@abstract_cluster.AbstractCluster.register_subclass("mp")
class MultiprocessingCluster(abstract_cluster.AbstractCluster):
    """
    MultiprocessingCluster
    """

    def __init__(self, conf_cluster, out_dir, launch_worker=True):
        """
        Init function of MultiprocessingCluster

        :param conf_cluster: configuration for cluster

        """

        self.out_dir = out_dir

        # Check conf
        checked_conf_cluster = self.check_conf(conf_cluster)

        # retrieve parameters
        self.nb_workers = checked_conf_cluster["nb_workers"]
        self.dump_to_disk = checked_conf_cluster["dump_to_disk"]
        self.per_job_timeout = checked_conf_cluster["per_job_timeout"]

        # Set multiprocessing mode
        # forkserver is used, to allow OMP to be used in numba
        mp_mode = "forkserver"

        self.launch_worker = launch_worker

        self.tmp_dir = None

        if self.launch_worker:
            # Create wrapper object
            if self.dump_to_disk:
                if self.out_dir is None:
                    raise Exception("Not out_dir provided")
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
                self.tmp_dir = os.path.join(self.out_dir, "tmp_save_disk")
                if not os.path.exists(self.tmp_dir):
                    os.makedirs(self.tmp_dir)
                self.wrapper = wrapper.WrapperDisk(self.tmp_dir)
            else:
                self.wrapper = wrapper.WrapperNone(None)

            # Create pool
            self.pool = mp.get_context(mp_mode).Pool(
                self.nb_workers,
                initializer=freeze_support,
                maxtasksperchild=100,
            )
            self.queue = Queue()
            self.task_cache = {}

            # Refresh worker
            self.refresh_worker = threading.Thread(
                target=MultiprocessingCluster.refresh_task_cache,
                args=(
                    self.pool,
                    self.task_cache,
                    self.queue,
                    self.per_job_timeout,
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
        # set OTB_MAX_RAM_HINT = total_ram / nb_worker or 4000 by default
        # set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = 1

        # Overload conf
        overloaded_conf["mode"] = conf.get("mode", "mp")
        nb_workers = conf.get("nb_workers", 2)
        overloaded_conf["nb_workers"] = min(available_cpu, nb_workers)
        overloaded_conf["dump_to_disk"] = conf.get("dump_to_disk", True)
        overloaded_conf["per_job_timeout"] = conf.get("per_job_timeout", 600)

        cluster_schema = {
            "mode": str,
            "dump_to_disk": bool,
            "nb_workers": int,
            "per_job_timeout": Or(float, int),
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

    def create_task(self, func, nout=1):
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

            used_func, used_kwargs = self.wrapper.get_function_and_kwargs(
                func, kwargs, nout=nout
            )

            # create delayed_task
            delayed_task = MpDelayedTask(used_func, list(argv), used_kwargs)

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

        def check_arg(obj, current_can_run):
            """
            Check if arg is a delayed.
            If obj is a delayed and job is done, replace it by MpJob
            And add it to arguments list

            :param obj: object to process
            :param current_can_run : copy of global canRun
            """
            new_can_run = current_can_run
            if isinstance(obj, MpDelayed):
                rec_future = self.rec_start(obj, memorize)
                new_obj = MpJob(
                    rec_future.mp_future_task.job_id, rec_future.return_index
                )
                new_can_run = False
            else:
                new_obj = obj
            return new_obj, new_can_run

        # inspect args recursively
        filt_args = []
        for idx, _ in enumerate(current_delayed_task.args):
            if isinstance(current_delayed_task.args[idx], list):
                current_idx_args_list = []
                for idx2 in range(len(current_delayed_task.args[idx])):
                    new_obj, can_run = check_arg(
                        current_delayed_task.args[idx][idx2], can_run
                    )
                    current_idx_args_list.append(new_obj)
                filt_args.append(current_idx_args_list)
            else:
                new_obj, can_run = check_arg(
                    current_delayed_task.args[idx], can_run
                )
                filt_args.append(new_obj)

        # inspect kwargs recursively
        filt_kw = {}
        for key in current_delayed_task.kw_args.keys():
            if isinstance(current_delayed_task.kw_args[key], list):
                current_idx_args_list = []
                for idx2 in range(len(current_delayed_task.kw_args[key])):
                    new_obj, can_run = check_arg(
                        current_delayed_task.kw_args[key][idx2], can_run
                    )
                    current_idx_args_list.append(new_obj)
                filt_kw[key] = current_idx_args_list
            else:
                new_obj, can_run = check_arg(
                    current_delayed_task.kw_args[key], can_run
                )
                filt_kw[key] = new_obj

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
        pool, task_cache, in_queue, per_job_timeout
    ):
        """
        Refresh task cache

        :param task_cache: task cache list
        :param in_queue: queue

        """
        thread = threading.current_thread()

        wait_list = {}
        in_progress_list = {}

        while thread._state == RUN:  # pylint: disable=W0212
            # wait before next iteration
            time.sleep(REFRESH_TIME)
            # get new task from queue
            if not in_queue.empty():
                # get all task from this batch
                for job_id, can_run, func, args, kw_args in iter(
                    in_queue.get, "END_BATCH"
                ):
                    if can_run:
                        in_progress_list[job_id] = pool.apply_async(
                            func, args=args, kwds=kw_args
                        )
                    else:
                        wait_list[job_id] = [func, args, kw_args]

            # check for ready results
            done_list = {}
            for job_id, job_id_progress in in_progress_list.items():
                if job_id_progress.ready():
                    try:
                        res = job_id_progress.get(timeout=per_job_timeout)
                        success = True
                    except Exception as exception:
                        res = exception
                        success = False
                    done_list[job_id] = [success, res]

            # clean done jobs
            for job_id, _ in done_list.items():
                del in_progress_list[job_id]

            # check wait_list for dependent tasks

            ready_list = []
            for job_id, wait_job_id in wait_list.items():
                func, args, kw_args = wait_job_id
                can_run = True
                for idx, _ in enumerate(args):
                    args[idx], can_run = check_job_done(
                        done_list, args[idx], can_run
                    )
                    if isinstance(args[idx], list):
                        for idx2 in range(len(args[idx])):
                            args[idx][idx2], can_run = check_job_done(
                                done_list, args[idx][idx2], can_run
                            )

                # inspect kwargs recursively
                for key in kw_args:
                    kw_args[key], can_run = check_job_done(
                        done_list, kw_args[key], can_run
                    )
                    if isinstance(kw_args[key], list):
                        for idx2 in range(len(kw_args[key])):
                            kw_args[key][idx2], can_run = check_job_done(
                                done_list, kw_args[key][idx2], can_run
                            )

                # mask as ready to run
                if can_run:
                    ready_list.append(job_id)

            # copy results to futures (they remove themselves from task_cache
            for job_id, done_job_id in done_list.items():
                task_cache[job_id].set(done_job_id)

            # launch tasks ready to run
            for job_id in ready_list:
                func, args, kw_args = wait_list[job_id]
                in_progress_list[job_id] = pool.apply_async(
                    func, args=args, kwds=kw_args
                )
                del wait_list[job_id]

    def future_iterator(self, future_list):
        """
        Start all tasks

        :param future_list: future_list list
        """

        return MpFutureIterator(future_list, self)


def check_job_done(done_list, obj, current_can_run):
    """
    Check if obj is a delayed.
    If obj is a delayed and job is done, replace it

    :param done_list: list of done tasks
    :param obj: object to process
    :param current_can_run: current global can_run
    """
    new_can_run = current_can_run
    new_obj = obj
    if isinstance(obj, MpJob):
        if obj.task_id in done_list:
            if not done_list[obj.task_id][0]:
                # Task ended with an error but we need the result
                # for a dependent task
                raise done_list[obj.task_id][1]

            if isinstance(done_list[obj.task_id][1], tuple):
                new_obj = done_list[obj.task_id][1][obj.r_idx]
            else:
                if obj.r_idx > 0:
                    raise ValueError("Asked for index > 0 in a singleton")
                new_obj = done_list[obj.task_id][1]

        else:
            new_can_run = False
    return new_obj, new_can_run


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


class MpDelayed:  # pylint: disable=R0903
    """
    multiprocessing version of dask.delayed
    """

    def __init__(self, delayed_task, return_index=0):
        self.delayed_task = delayed_task
        self.return_index = return_index

        # register to delayed_task
        self.delayed_task.associated_objects.append(self)


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
                    res = item
                    break

        self.future_list.remove(res)
        return self.cluster.wrapper.get_obj(res.get())

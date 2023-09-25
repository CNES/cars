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
Contains functions needed to factorize delayed
"""

# CARS imports
from cars.orchestrator.cluster.mp_cluster.mp_objects import (
    FactorizedObject,
    MpDelayedTask,
)


def factorize_delayed(task_list):
    """
    Factorize Task list

    Only factorize tasks that depends on a single task

    :param task_list: list of delayed
    :type task_list: list(MpDelayed)

    """
    # Compute graph usage
    graph_usages = compute_graph_delayed_usages(task_list)

    # Factorize delayed
    already_seen_delayed_tasks = []
    for delayed in task_list:
        factorize_delayed_rec(delayed, graph_usages, already_seen_delayed_tasks)


def factorize_delayed_rec(delayed, graph_usages, already_seen_delayed_tasks):
    """
    Factorize Task list

    Only factorize tasks that depends on a single task

    :param delayed: delayed to factorize
    :type delayed: MpDelayed
    :param graph_usages: number of usages of delayed
    :type graph_usages: dict
        example: {delayed1: 3}
    :param already_seen_delayed_tasks: list of MpDelayedTask already seen

    """

    # check if current delayed can be factorized
    depending_delayed = delayed.get_depending_delayed()

    number_depending_task = compute_nb_depending_task(depending_delayed)
    max_nb_of_usages = 0
    if len(depending_delayed) > 0:
        max_nb_of_usages = max(
            number_of_usage(deld, graph_usages) for deld in depending_delayed
        )

    current_task = delayed.delayed_task

    if current_task not in already_seen_delayed_tasks:
        if number_depending_task == 1 and max_nb_of_usages == 1:
            previous_task = depending_delayed[0].delayed_task
            factorized_object = FactorizedObject(current_task, previous_task)

            # Create new task and assign it to current delay
            new_task = MpDelayedTask(factorized_fun, [factorized_object], {})
            new_task.associated_objects = current_task.associated_objects
            delayed.delayed_task = new_task

            # Factorize again with current
            factorize_delayed_rec(
                delayed, graph_usages, already_seen_delayed_tasks
            )

        else:
            # Only set to seen when task is completly factorized
            already_seen_delayed_tasks.append(current_task)
            # Get new dependances and factorize it
            depending_delayed = delayed.get_depending_delayed()
            for new_delayed in depending_delayed:
                factorize_delayed_rec(
                    new_delayed, graph_usages, already_seen_delayed_tasks
                )


def compute_graph_delayed_usages(task_list):
    """
    Compute the number of times every delayed is used in graph

    :param task_list: list of delayed
    :type task_list: list(MpDelayed)

    :return: number of usages of delayed
    :rtype: dict
        example: {delayed1: 3}
    """

    graph_usages = {}
    already_seen_tasks = []

    for delayed in task_list:
        get_delayed_usage_rec(delayed, graph_usages, already_seen_tasks)

    return graph_usages


def get_delayed_usage_rec(delayed, graph_usages, already_seen_tasks):
    """
    Get number of time input delayed is used

    :param delayed: delayed to factorize
    :type delayed: MpDelayed
    :param graph_usages: number of usages of delayed
    :type graph_usages: dict
        example: {delayed1: 3}
    :param already_seen_tasks: list of seen delayed task
    :type already_seen_tasks: list[MpDelayedTask]
    """

    # update graph_usages
    if delayed in graph_usages:
        graph_usages[delayed] += 1
    else:
        graph_usages[delayed] = 1

    # get usage of task inputs, if task was not already seen
    delayed_task = delayed.delayed_task

    if delayed_task not in already_seen_tasks:
        # add task to seen
        already_seen_tasks.append(delayed_task)

        # get usage for all inputs
        depending_delayed = delayed.get_depending_delayed()
        for input_delayed in depending_delayed:
            get_delayed_usage_rec(
                input_delayed, graph_usages, already_seen_tasks
            )


def number_of_usage(delayed, graph_usages):
    """
    Compute the number of times a delayed is used

    :param delayed: delayed to factorize
    :type delayed: MpDelayed
    :param graph_usages: number of usages of delayed
    :type graph_usages: dict
        example: {delayed1: 3}

    :return: number of usages of delayed
    :rtype: int
    """

    nb_usage = graph_usages[delayed]
    return nb_usage


def compute_nb_depending_task(depending_delayed_list):
    """
    Compute the number of different delayed task in list of delayed

    :param depending_delayed_list: list of delayed
    :type depending_delayed_list: list[MpDelayed]

    :return: number of depending task
    :rtype: int
    """

    list_delayed_task = []

    for delayed in depending_delayed_list:
        delayed_task = delayed.delayed_task
        if delayed_task not in list_delayed_task:
            list_delayed_task.append(delayed_task)

    return len(list_delayed_task)


# Factorized function and its generator
def factorized_fun(factorized_object):
    """
    This function unpack multiple functions with their arguments,
    and run them sequentialy until task list is empty

    :param factorized_object: Object that contains a list of tasks
    :type factorized_object: mp_objects.FactorizedObject
    """

    res = None
    while factorized_object.tasks:
        # Run next task with output of previous task
        res = factorized_object.pop_next_task(previous_result=res)
    return res

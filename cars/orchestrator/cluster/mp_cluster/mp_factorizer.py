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
# No longer supported
# TODO remove

import copy

# CARS imports
from cars.orchestrator.cluster.mp_cluster.mp_objects import MpDelayed


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
            # We can factorize current delayed with next ones

            #  modify delayed task

            # Get current params
            # Here we only have one depending task, we can fuse them

            # Modify delayed to "POS"
            current_fun = current_task.func
            current_args = current_task.args
            current_kwargs = current_task.kw_args
            for index, arg in enumerate(current_args):
                if isinstance(arg, MpDelayed):
                    current_args[index] = "POS_" + repr(arg.return_index)
            for index, kw_arg in enumerate(current_kwargs):
                if isinstance(kw_arg, MpDelayed):
                    current_kwargs[index] = "POS_" + repr(kw_arg.return_index)

            # previous_params
            previous_task = depending_delayed[0].delayed_task
            previous_fun = previous_task.func
            previous_args = previous_task.args
            previous_kwargs = previous_task.kw_args

            # compute new params
            new_fun, new_args, new_kwargs = generate_args_kwargs_factorize(
                previous_fun,
                previous_args,
                previous_kwargs,
                current_fun,
                current_args,
                current_kwargs,
            )

            # Modify current task
            # Task is only used here, nothing else is modified in graph
            current_task.func = new_fun
            current_task.args = new_args
            current_task.kw_args = new_kwargs

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
        # + 1
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


def factorized_fun(*args, **kwargs):
    """
    This function unpack multiple functions with their arguments,
    and run them sequentialy


    """

    # get keys
    # Get fun, args, and clean kwargs
    next_fun = {}
    next_funs_keys = []

    current_kwargs = copy.copy(kwargs)

    for key, key_item in kwargs.items():
        if key.startswith("NEXT_FUN_"):
            next_funs_keys.append(key)
            next_fun[key] = key_item
            current_kwargs.pop(key)

    fun_0 = next_fun["NEXT_FUN_0"]["fun"]

    # run first function
    res = fun_0(*args, **current_kwargs)

    # run other functions
    for i in range(1, len(next_funs_keys)):  # pylint: disable=C0200
        current_key = "NEXT_FUN_" + repr(i)

        current_fun = next_fun[current_key]["fun"]
        current_args = next_fun[current_key]["args"]
        current_kwargs = next_fun[current_key]["kwarg"]

        # replace args computed by previous run
        for iter_args in range(len(current_args)):  # pylint: disable=C0200
            if isinstance(current_args[iter_args], str):
                if "POS_" in current_args[iter_args]:
                    pos = int(current_args[iter_args].split("_")[1])

                    if isinstance(res, tuple):
                        current_args[iter_args] = res[pos]
                    else:
                        if pos != 0:
                            raise RuntimeError(
                                "waiting multiple output but res is not tuple"
                            )
                        current_args[iter_args] = res

        # run function
        res = current_fun(*current_args, **current_kwargs)

    return res


def get_number_of_steps(kwargs):
    """
    Get number of following function stored in kwargs

    :param kwargs: key arguments of function
    :type kwargs: dict

    :return: number of steps
    :rtype: int
    """

    count = 0
    for key in kwargs:
        if key.startswith("NEXT_FUN_"):
            count += 1

    return count


def generate_args_kwargs_factorize(fun1, args1, kwargs1, fun2, args2, kwargs2):
    """
    Generate args for new delayed

    WARNING : modify inplace args and kwargs
    if you compute inbetween the function errors will occure

    :param fun1: first function to be called
    :type fun1: callable
    :param args1: list of arguments of function 1
    :type args1: list
    :param kwargs1: dict of keyword arguments of function 1
    :type kwargs1: dict
    :pram fun2: second function to be called
    :type fun2: callable
    :param args2: list of arguments of function 2
    :type args2: list
    :param kwargs2: dict of keyword arguments of function 2
    :type kwargs2: dict

    :return: new function, new arguments, new keyword arguments
    :rtype: function, list, dict
    """

    # fun2 can already be a factorized fun

    new_fun = factorized_fun

    # Generate fist
    first_addon = {"fun": fun1, "args": args1, "kwarg": kwargs1}

    # Generate second
    second_addon = {"fun": fun2, "args": args2, "kwarg": kwargs2}

    # Check if one of the delayed is already factorized
    if factorized_fun not in (fun1, fun2):  # pylint: disable=W0143
        # Addon will only contain f1, the rest is in args and kwargs
        # for input transformation purposes
        new_args = args1
        new_kwargs = kwargs1
        first_addon.pop("args")
        first_addon.pop("kwarg")

        # Add addon in kwargs
        new_kwargs["NEXT_FUN_0"] = first_addon
        new_kwargs["NEXT_FUN_1"] = second_addon

    else:
        if fun1 == factorized_fun:  # pylint: disable=W0143
            # f2 will be added to the end
            new_args = args1
            new_kwargs = kwargs1
            nb_factorized = get_number_of_steps(kwargs1)
            # remove "NEXT_FUN" in second_addon
            # for step in range(0, nb_factorized):
            #    second_addon["kwarg"].pop("NEXT_FUN_" + repr(step))
            new_kwargs["NEXT_FUN_" + repr(nb_factorized)] = second_addon

        else:
            # f2 is factorized
            # f1 will be position 0, the rest will translate to 1
            new_args = args1
            new_kwargs = kwargs1
            first_addon.pop("args")
            first_addon.pop("kwarg")

            nb_factorized = get_number_of_steps(kwargs2)
            # translate functions
            for step in range(nb_factorized, 0, -1):
                kwargs1["NEXT_FUN_" + repr(step)] = kwargs2[
                    "NEXT_FUN_" + repr(step - 1)
                ]
                kwargs2.pop("NEXT_FUN_" + repr(step - 1))

            # 1 was 0 in factorized 2
            kwargs1["NEXT_FUN_" + repr(1)]["args"] = args2
            kwargs1["NEXT_FUN_" + repr(1)]["kwarg"] = kwargs2

            # add addon 1 as first
            new_kwargs["NEXT_FUN_0"] = first_addon

    return new_fun, new_args, new_kwargs

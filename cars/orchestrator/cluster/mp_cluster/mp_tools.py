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
Contains tools for multiprocessing
"""


def replace_data_rec(list_or_dict, func_to_apply, *func_args):
    """
    Replace MpJob in list or dict by their real data recursively

    :param list_or_dict: list or dict of data
    :param func_to_apply: function to apply
    :param func_args: function arguments

    :return: list or dict with real data
    :rtype: list, tuple, dict
    """

    if isinstance(list_or_dict, (list, tuple)):
        res = []
        for arg in list_or_dict:
            if isinstance(arg, (list, tuple, dict)):
                res.append(replace_data_rec(arg, func_to_apply, *func_args))
            else:
                res.append(func_to_apply(arg, *func_args))
        if isinstance(list_or_dict, tuple):
            res = tuple(res)

    elif isinstance(list_or_dict, dict):
        res = {}
        for key, value in list_or_dict.items():
            if isinstance(value, (list, dict, tuple)):
                res[key] = replace_data_rec(value, func_to_apply, *func_args)
            else:
                res[key] = func_to_apply(value, *func_args)
    else:
        raise TypeError("Function only support list or dict or tuple")

    return res

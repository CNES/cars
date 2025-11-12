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
this module contains tools checking bulldozer parameter used for memory
"""


import shutil


def can_allocate_shared_memory(nb_go=0.8):
    """
    Check if can allocate shared memory
    """
    # TODO remove when Eoascale not in bulldozer anymore
    try:
        # 1 Go = 1024 * 1024 * 1024 octets
        shm_stats = shutil.disk_usage("/dev/shm")
        available_bytes = shm_stats.free
        required_bytes = nb_go * 1024 * 1024 * 1024

        if available_bytes > required_bytes:
            log_message = "Can allocate shared memory."
            return True, log_message

        log_message = (
            "Cannot allocate shared memory: {} Go available"
            "If CARS runs on docker, use --shm-size option "
            "on docker run command. For instance: --shm-size=10Go".format(
                int(available_bytes / (1024 * 1024 * 1024))
            )
        )
        return False, log_message
    except Exception:
        # Possibly on windows
        log_message = "Crash on shared memory check."
        return True, log_message

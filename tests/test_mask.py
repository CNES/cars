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

import pytest

from utils import absolute_data_path
from cars import mask


@pytest.mark.unit_tests
def test_mask_classes_can_open():
    mask_classes_path = absolute_data_path("input/phr_paca/left_msk_classes.json")
    assert mask.mask_classes_can_open(mask_classes_path) is True

    wrong_mask_classes_path = absolute_data_path("input/mask_input/msk_wrong_json.json")
    assert mask.mask_classes_can_open(wrong_mask_classes_path) is False


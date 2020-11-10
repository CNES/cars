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
import numpy as np

from utils import absolute_data_path
from cars import mask_classes


@pytest.mark.unit_tests
def test_mask_classes_can_open():
    mask_classes_path = absolute_data_path("input/phr_paca/left_msk_classes.json")
    assert mask_classes.mask_classes_can_open(mask_classes_path) is True

    wrong_mask_classes_path = absolute_data_path("input/mask_input/msk_wrong_json.json")
    assert mask_classes.mask_classes_can_open(wrong_mask_classes_path) is False


@pytest.mark.unit_tests
def test_carsmask_is_multiclasses_mask():
    mc_msk = np.array([[mask_classes.VALID_VALUE, mask_classes.VALID_VALUE, 2],
                       [1,                        mask_classes.VALID_VALUE, 100],
                       [mask_classes.VALID_VALUE, 100,                      200]])

    is_mc_mask = mask_classes.is_multiclasses_mask(mc_msk)

    assert is_mc_mask is True

    not_mc_msk = np.array([[mask_classes.VALID_VALUE, mask_classes.VALID_VALUE, mask_classes.NO_DATA_IN_EPIPOLAR_RECTIFICATION],
                           [1,                        mask_classes.VALID_VALUE, 1],
                           [mask_classes.VALID_VALUE, 1,                        1]], dtype=np.uint16)

    is_mc_mask = mask_classes.is_multiclasses_mask(not_mc_msk)

    assert is_mc_mask is False

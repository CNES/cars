#!/bin/bash
##
## Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
##
## This file is part of CARS
## (see https://github.com/CNES/cars).
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

export CARSPATH="$(dirname $( cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P ))"

PYVERSION="$(python --version)"
PYVERSION=${PYVERSION#* }
PYVERSION=${PYVERSION%.*}

export PATH=$PATH:$CARSPATH/bin
export CARS_STATIC_CONFIGURATION=$CARSPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CARSPATH/lib
export OTB_APPLICATION_PATH=$OTB_APPLICATION_PATH:$CARSPATH/lib/python${PYVERSION}/site-packages/otb/applications/

# environment variables that affect Orfeo ToolBox
export OTB_GEOID_FILE="GEOID_FILE_TO_REPLACE"
export CARS_NB_WORKERS_PER_PBS_JOB=2
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
export OPJ_NUM_THREADS=$OMP_NUM_THREADS
export GDAL_NUM_THREADS=$OMP_NUM_THREADS
export OTB_MAX_RAM_HINT=2000
export OTB_LOGGER_LEVEL=WARNING
export GDAL_CACHEMAX=128
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export CARS_TEST_TEMPORARY_DIR=$HOME

# virtualenv + autocompletion
eval "$(register-python-argcomplete cars_cli)"

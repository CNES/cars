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

# Get root path of cars and env_cars command (one level down)
export ROOT_PATH="$(dirname $( cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P ))"

# CARS OTB applications PATH
# Because link to installed libs doesn't always work with setuptools, link to build directory
export OTB_APPLICATION_PATH=$OTB_APPLICATION_PATH:$ROOT_PATH/../build/lib/otb/applications/
# Complement with link to installed libs path (in sys.prefix of venv or main standard root)
export OTB_APPLICATION_PATH=$OTB_APPLICATION_PATH:$ROOT_PATH/lib/

# add PATH and LD_LIBRARY_PATH for CARS OTB added apps to work in shell
export PATH=$PATH:$ROOT_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_PATH/lib #

## Environment variables that affect Orfeo ToolBox and co external libs
# OTB
export OTB_LOGGER_LEVEL=WARNING # Set OTB log level to limit OTB verbosity
# OpenMP
export OMP_NUM_THREADS=1 # Set OpenMP threads for CARS and DASK
# OpenJpeg
export OPJ_NUM_THREADS=$OMP_NUM_THREADS # Set OpenJpeg threads as OpenMP
# GDAL
export GDAL_NUM_THREADS=$OMP_NUM_THREADS # Set GDAL threads as OpenMP
export GDAL_CACHEMAX=128 # Set GDAL cache max size
# ITK
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 #set default ITK threads

# NUMBA configuration
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS # Set Numba threads as OpenMP

# CARS PBS DASK Number of workers per pbs job
# TODO: evolution with cluster refactoring, remove env
export CARS_NB_WORKERS_PER_PBS_JOB=2

## CARS autocompletion

# Define command exists function
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

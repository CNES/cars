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

# Install CARS, whether via
#      ``python setup.py install``
#    or
#      ``pip install cars``

import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

import sys
from distutils.spawn import find_executable, spawn
import re
from pathlib import Path


# Meta-data.
NAME = 'cars'
DESCRIPTION = 'CARS is a multi-view stereovision pipeline for satellite images. It produces Digital Surface Model in raster format.'
URL = 'https://github.com/CNES/cars'
AUTHOR = 'CNES'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.0'
EMAIL = 'david.youssefi@cnes.fr'
LICENSE = 'Apache License 2.0'
REQUIREMENTS = ['numpy>=1.17.0',
                'scipy',
                'matplotlib',
                'affine',
                'rasterio==1.1.1',
                'dask',
                'distributed',
                'dask-jobqueue',
                'jupyter',
                'bokeh',
                'pylint',
                'pytest',
                'pytest-cov',
                'json-checker',
                'xarray',
                'k3d',
                'tqdm',
                'sphinx-rtd-theme',
                'netCDF4==1.5.3',
                'GitPython',
                'argcomplete',
                'Shapely',
                'Fiona',
                'pyproj',
                'numba',
                'pandas',
                'tbb',
                'pandora-plugin-libsgm==0.2.2']

class CustomBuildPyCommand(build_py):
    """
       Add custom step for CARS installation
    """

    def check(self):
        """
           Check environment
           Test if Geoid file is provided
        """
        if os.environ.get("OTB_GEOID_FILE") is None:
            raise Exception("OTB_GEOID_FILE not set")

    def create_env_script(self):
        """
           Create environment file to set CARS environment
        """
        GEOID_FILE_TO_REPLACE = os.environ.get("OTB_GEOID_FILE")
        template = None 
        with open('template_env_cars.sh') as src: 
             template = src.readlines()
        with open('env_cars.sh','w') as dest: 
             for line in template: 
                 if "GEOID_FILE_TO_REPLACE" in line: 
                     dest.write(re.sub(r'GEOID_FILE_TO_REPLACE', GEOID_FILE_TO_REPLACE, line)) 
                 else: 
                     dest.write(line)

    def run(self):
        self.check()
        self.create_env_script()
        super(CustomBuildPyCommand,self).run()


class CompileInstallCommand(install):
    """
       Add custom step for CARS installation
    """

    def check(self):
        """
           Check environment
           Test requirements for build : cmake, OTB, vlfeat
           Test if Geoid file is provided
        """
        if find_executable("cmake") is None:
            raise Exception("Command cmake not found")
        if find_executable("otbcli_ReadImageInfo") is None:
            raise Exception("OTB not found")
        if os.environ.get("OTB_APPLICATION_PATH") is None:
            raise Exception("OTB_APPLICATION_PATH not set")
        if os.environ.get("VLFEAT_INCLUDE_DIR") is None:
            raise Exception("VLFEAT_INCLUDE_DIR not set")


    def compile_and_install_cars(self):
        """
           Run external script to comile et install OTB remote modules
        """
        if sys.prefix is None:
            sys.prefix = "/usr/local"
        current_dir = os.getcwd()
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        os.chdir("build")
        cmd = ['cmake',
               f'-DCMAKE_INSTALL_PREFIX={sys.prefix}',
               '-DOTB_BUILD_MODULE_AS_STANDALONE=ON',
               '-DCMAKE_BUILD_TYPE=Release',
               f'-DVLFEAT_INCLUDE_DIR={os.environ["VLFEAT_INCLUDE_DIR"]}',
               '../otb_remote_module']
        spawn(cmd)
        spawn(['make','install'])
        os.chdir(current_dir)


    def run(self):
        self.check()
        self.compile_and_install_cars()
        super(CompileInstallCommand,self).run()


# Setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    packages=find_packages(),
    data_files=[('static_conf', ['static_conf/static_configuration.json'])],
    long_description=DESCRIPTION,
    install_requires=REQUIREMENTS,
    python_requires=REQUIRES_PYTHON,
    entry_points={
                  'console_scripts': ['cars_cli = bin.cars_cli:entry_point']
                 },
    cmdclass={
              'build_py': CustomBuildPyCommand,
              'install': CompileInstallCommand
             },
    scripts=['env_cars.sh']
)

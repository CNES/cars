#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
CARS setup.py
Most of the configuration is in setup.cfg except :
  - Surcharging subcommand install and develop with cars OTB build
  - Enabling setuptools_scm
"""
import os
import sys
from pathlib import Path
from shutil import which
from subprocess import run

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def cars_otb_build(command_subclass):
    """
    A decorator subclassing one of the setuptools commands.
    It modifies the run() method
    """
    orig_run = command_subclass.run

    def cars_check_env():
        """
        Check environment
        Test requirements for build : cmake, OTB, vlfeat
        """
        if which("cmake") is None:
            raise Exception("Command cmake not found")
        if which("otbcli_ReadImageInfo") is None:
            raise Exception("OTB not found")
        if os.environ.get("OTB_APPLICATION_PATH") is None:
            raise Exception("OTB_APPLICATION_PATH not set")
        if os.environ.get("VLFEAT_INCLUDE_DIR") is None:
            raise Exception("VLFEAT_INCLUDE_DIR not set")

    def cars_otb_build_install():
        """
        Run external script to build and install OTB remote modules
        """
        if sys.prefix is None:
            sys.prefix = "/usr/local"
        current_dir = os.getcwd()
        Path("build").mkdir(exist_ok=True)
        os.chdir("build")
        # Build cmake cmd for OTB applications
        cmd = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={sys.prefix}",
            "-DOTB_BUILD_MODULE_AS_STANDALONE=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f'-DVLFEAT_INCLUDE_DIR={os.environ["VLFEAT_INCLUDE_DIR"]}',
            "../otb_remote_module",
        ]
        run(cmd, check=True)
        # Install OTB applications in sys.prefix/lib (path in env_cars.sh)
        run(["make", "install"], check=True)
        os.chdir(current_dir)

    def modified_run(self):
        print("CARS check environment")
        cars_check_env()
        print("CARS OTB modules build and install ")
        cars_otb_build_install()
        # Launch original subcommand run()
        orig_run(self)

    command_subclass.run = modified_run
    return command_subclass


# Apply same cars specific setup decorator to custom setup commands:
#   develop : pip install -e .
#   install : pip install .
@cars_otb_build
class CustomDevelopCommand(develop):
    pass


@cars_otb_build
class CustomInstallCommand(install):
    pass


# Setup with setup.cfg config
setup(
    use_scm_version=True,
    cmdclass={"install": CustomInstallCommand, "develop": CustomDevelopCommand},
)

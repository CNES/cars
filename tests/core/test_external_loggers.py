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
Test module for cars_logging with external loggers
Ensures that external loggers continue to work after CARS setup_logging calls
"""

import logging
import os
import tempfile
from io import StringIO

import pytest

from cars.core import cars_logging


class TestExternalLoggers:
    """
    Test that external loggers are not affected by CARS logging setup
    """

    def teardown_method(self):
        """Clean up handlers after each test"""
        # Remove all handlers from CARS logger
        cars_logger = logging.getLogger("CARS")
        for handler in cars_logger.handlers[:]:
            cars_logger.removeHandler(handler)
            handler.close()

    @pytest.mark.unit_tests
    def test_external_logger_survives_cars_setup_global(self):
        """
        Test that an external logger continues to work
        after setup_logging_global() is called.
        """
        # Create an external logger (simulating external tool)
        external_logger = logging.getLogger("ExternalTool")

        # Create a string stream handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        external_logger.addHandler(handler)
        external_logger.setLevel(logging.INFO)

        # Test 1: External logger works before CARS setup
        external_logger.info("Before CARS setup")
        output_before = stream.getvalue()
        assert (
            "ExternalTool - INFO - Before CARS setup" in output_before
        ), "External logger should work before CARS setup"

        # Setup CARS logging
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="INFO",
                global_log_file=log_file,
                use_stdout=False,
            )

            # Clear stream for next test
            stream.truncate(0)
            stream.seek(0)

            # Test 2: External logger still works after CARS setup
            external_logger.info("After CARS setup")
            output_after = stream.getvalue()
            assert (
                "ExternalTool - INFO - After CARS setup" in output_after
            ), "External logger should still work after CARS setup"

            # Test 3: CARS logger also works
            cars_logging.logger.info("CARS message")

    @pytest.mark.unit_tests
    def test_external_logger_survives_cars_setup_pipeline(self):
        """
        Test that an external logger continues to work
        after setup_logging_pipeline() is called.
        """
        # Create an external logger
        external_logger = logging.getLogger("ExternalService")

        # Create a string stream handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        external_logger.addHandler(handler)
        external_logger.setLevel(logging.DEBUG)

        # Setup global CARS logging first
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="INFO",
                global_log_file=log_file,
                use_stdout=False,
            )

            # Test external logger before pipeline setup
            external_logger.info("Before pipeline setup")
            output_before = stream.getvalue()
            assert (
                "ExternalService - INFO - Before pipeline setup"
                in output_before
            )

            # Setup pipeline logging
            log_dir = os.path.join(tmpdir, "logs")
            cars_logging.setup_logging_pipeline(
                loglevel="INFO",
                out_dir=log_dir,
                pipeline="test_pipeline",
            )

            # Clear stream
            stream.truncate(0)
            stream.seek(0)

            # Test external logger after pipeline setup
            external_logger.info("After pipeline setup")
            output_after = stream.getvalue()
            assert (
                "ExternalService - INFO - After pipeline setup" in output_after
            ), "External logger should still work after pipeline setup"

    @pytest.mark.unit_tests
    def test_external_logger_survives_cars_setup_workers(self):
        """
        Test that an external logger continues to work
        after setup_logging_workers() is called.
        """
        # Create an external logger
        external_logger = logging.getLogger("ExternalWorkerTool")

        # Create a string stream handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        external_logger.addHandler(handler)
        external_logger.setLevel(logging.DEBUG)

        # Setup global CARS logging first
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="INFO",
                global_log_file=log_file,
                use_stdout=False,
            )

            # Setup workers logging
            log_dir = os.path.join(tmpdir, "logs", "workers_log")
            cars_logging.setup_logging_workers(
                loglevel="DEBUG",
                log_dir=log_dir,
            )

            # Test external logger after workers setup
            external_logger.info("After workers setup")
            output = stream.getvalue()
            assert (
                "ExternalWorkerTool - INFO - After workers setup" in output
            ), "External logger should still work after workers setup"

    @pytest.mark.unit_tests
    def test_external_logger_handlers_not_affected(self):
        """
        Test that external loggers' handlers are not affected by CARS setup
        """
        external_logger = logging.getLogger("ExternalHandler")

        # Add multiple handlers to external logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        external_logger.addHandler(handler)
        external_logger.setLevel(logging.INFO)

        initial_handler_count = len(external_logger.handlers)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup CARS logging
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="INFO",
                global_log_file=log_file,
                use_stdout=False,
            )

            # External logger handlers should not change
            final_handler_count = len(external_logger.handlers)
            assert (
                initial_handler_count == final_handler_count
            ), "CARS setup should not add/remove handlers from external loggers"

            # Handler should still work
            external_logger.info("test")
            assert "ExternalHandler - test" in stream.getvalue()

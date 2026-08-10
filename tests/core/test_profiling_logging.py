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
Test module for cars_logging profiling functionality
Ensures that profiling messages are correctly logged to profiling files
and not to standard output/info logs
"""

import logging
import os
import tempfile

import pytest

from cars.core import cars_logging


class TestProfilingLogging:
    """
    Test that profiling messages are correctly handled
    """

    def teardown_method(self):
        """Clean up handlers after each test"""
        # Remove all handlers from CARS logger
        cars_logger = logging.getLogger("CARS")
        for handler in cars_logger.handlers[:]:
            cars_logger.removeHandler(handler)
            handler.close()

    @pytest.mark.unit_tests
    def test_profiling_messages_in_profiling_file(self):
        """
        Test that profiling messages are written to profiling.log
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup global logging
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="DEBUG",
                global_log_file=log_file,
                use_stdout=False,
            )

            # Setup pipeline logging
            log_dir = os.path.join(tmpdir, "logs")
            cars_logging.setup_logging_pipeline(
                loglevel="DEBUG",
                out_dir=log_dir,
                pipeline="profiling_test",
            )

            # Send profiling message
            profiling_msg = "CarsProfiling%test_func%1.0%5.2%100.0%512%1024%50"
            cars_logging.add_profiling_message(profiling_msg)

            # Send regular messages
            cars_logging.logger.debug("Debug message")
            cars_logging.logger.info("Info message")
            cars_logging.logger.warning("Warning message")

            # Check profiling file exists and contains profiling message
            profiling_file = os.path.join(log_dir, "profiling", "profiling.log")
            assert os.path.exists(profiling_file), "Profiling file should exist"

            with open(profiling_file, "r", encoding="utf-8") as f:
                profiling_content = f.read()

            assert (
                profiling_msg in profiling_content
            ), "Profiling message should be in profiling.log"
            assert (
                "Debug message" not in profiling_content
            ), "Debug message should not be in profiling.log"
            assert (
                "Info message" not in profiling_content
            ), "Info message should not be in profiling.log"
            assert (
                "Warning message" not in profiling_content
            ), "Warning message should not be in profiling.log"

    @pytest.mark.unit_tests
    def test_worker_handlers_filter_correctly(self):
        """
        Test that WorkerHandler filters profiling and non-profiling correctly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup global and workers logging
            log_file = os.path.join(tmpdir, "cars.log")
            cars_logging.setup_logging_global(
                loglevel="INFO",
                global_log_file=log_file,
                use_stdout=False,
            )

            log_dir = os.path.join(tmpdir, "logs", "workers_log")
            cars_logging.setup_logging_workers(
                loglevel="DEBUG",
                log_dir=log_dir,
            )

            # Send profiling message
            cars_logging.add_profiling_message("CarsProfiling%worker_task%1.0")
            # Send regular message
            cars_logging.logger.info("Worker regular message")

            # Check workers.log contains regular but not profiling
            workers_log = os.path.join(log_dir, "workers.log")
            with open(workers_log, "r", encoding="utf-8") as f:
                workers_content = f.read()

            assert (
                "Worker regular message" in workers_content
            ), "Regular messages should be in workers.log"
            # Profiling messages should NOT be in workers.log
            # (they go to profiling.log instead)
            assert (
                "CarsProfiling%worker_task" not in workers_content
            ), "Profiling messages should not be in workers.log"

            # Check profiling.log contains profiling
            profiling_log = os.path.join(log_dir, "profiling.log")
            with open(profiling_log, "r", encoding="utf-8") as f:
                profiling_content = f.read()

            assert (
                "CarsProfiling%worker_task" in profiling_content
            ), "Profiling messages should be in profiling.log"

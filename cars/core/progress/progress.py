#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
"""Progress tree with Rich UI display.

One progress bar is shown per pipeline via Rich.
Tasks are registered under a pipeline with weights, and task
progress contributes to the pipeline bar accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count

from rich.console import Console

from cars.core.cars_logging import (
    add_progress_message,
    get_warning_count,
)
from cars.core.progress.ui import PipelineTreeUI


@dataclass
class TaskState:  # pylint: disable=too-many-instance-attributes
    """Internal state for one tracked task."""

    task_id: int
    name: str
    pipeline_id: int
    weight: float
    expected_runs: int = 1
    started_runs: int = 0
    nominal_started_runs: int = 0
    progress_in_run: float = 0.0
    total: int = 0
    last_logged_percent: int = 0  # for logging progress at intervals
    progressed_tiles: int = 0
    retries: int = 0
    failed: int = 0
    pending_retry_pass: bool = False
    current_pass_is_retry: bool = False
    completed_nominal_runs: int = 0


@dataclass
class PipelineState:
    """Internal state for one pipeline."""

    pipeline_id: int
    name: str
    position: int
    total_weight: float = 0.0
    weighted_progress: float = 0.0
    retries: int = 0
    failed: int = 0


class ProgressTree:
    """Singleton registry with pipeline-level weighted progress bars."""

    _instance: ProgressTree | None = None
    _initialized: bool = False

    def __new__(cls) -> ProgressTree:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._pipeline_id_gen = count(1)
        self._task_id_gen = count(1)
        self._pipeline_position_gen = count(0)
        self._pipelines: dict[int, PipelineState] = {}
        self._tasks: dict[int, TaskState] = {}
        self._task_to_pipeline: dict[int, int] = {}
        self._pipeline_parent: dict[int, int | None] = {}
        self._pipeline_children: dict[int, list[int]] = {}
        self._ui = PipelineTreeUI(console=Console())
        self._ui_enabled = True
        self._pipeline_order: list[int] = []
        self._initialized = True
        self._finished_successfully = False

    def set_ui_enabled(self, enabled: bool) -> None:
        """Enable/disable Rich UI rendering for progress updates."""
        self._ui_enabled = bool(enabled)
        if not self._ui_enabled and self._ui.live is not None:
            self._ui.live.stop()
            self._ui.live = None

    def update_log_file_path(self, log_file_path: str | None) -> None:
        """Update log file path used by the footer log link button."""
        self._ui.update_log_file_path(log_file_path)

    def update_empty_status_text(self, status_text: str) -> None:
        """Update startup status, shown before any pipeline is registered."""
        self._ui.update_empty_status_text(status_text)
        if self._ui_enabled:
            self.draw()

    def begin_pipeline(
        self, pipeline_name: str, parent_id: int | None = None
    ) -> int:
        """Declare that subsequent tasks belong to this pipeline."""
        pipeline_id = next(self._pipeline_id_gen)
        position = next(self._pipeline_position_gen)
        if parent_id is None:
            self._pipeline_order.append(pipeline_id)
        indent = 0 if parent_id is None else 1
        if self._ui_enabled:
            self._ui.add_node(
                pipeline_id,
                pipeline_name,
                indent=indent,
                parent_id=parent_id,
            )
            self._ui.update_state(pipeline_id, "pending")
            self._ui.update_warning_count(get_warning_count())
        self._pipelines[pipeline_id] = PipelineState(
            pipeline_id=pipeline_id,
            name=pipeline_name,
            position=position,
        )
        self._pipeline_parent[pipeline_id] = parent_id
        self._pipeline_children[pipeline_id] = []
        if parent_id is not None:
            self._pipeline_children.setdefault(parent_id, []).append(
                pipeline_id
            )
        return pipeline_id

    def register_task(
        self,
        pipeline_id: int,
        task_name: str,
        *,
        weight: float = 1.0,
        expected_runs: int = 1,
    ) -> int:
        """Register a task under a pipeline and return its unique ID."""
        if pipeline_id not in self._pipelines:
            raise ValueError(f"Unknown pipeline_id: {pipeline_id}")

        task_id = next(self._task_id_gen)
        weight = float(weight)
        expected_runs = max(1, int(expected_runs))
        if weight <= 0:
            logging.warning(
                f"ProgressTree warning: task {task_name} has "
                f"non-positive weight {weight}"
            )
        self._tasks[task_id] = TaskState(
            task_id=task_id,
            name=task_name,
            pipeline_id=pipeline_id,
            weight=weight,
            expected_runs=expected_runs,
        )
        self._task_to_pipeline[task_id] = pipeline_id
        self._pipelines[pipeline_id].total_weight += weight

        add_progress_message(
            "Registered task {} under pipeline {} with weight {}".format(
                task_name, self._pipelines[pipeline_id].name, weight
            )
        )
        return task_id

    def _refresh_pipeline_bar(self, pipeline: PipelineState) -> None:
        """
        Update Rich UI display for one pipeline
        from weighted progress and retry metadata.
        """
        total_progress, total_weight = self._aggregate_pipeline_progress(
            pipeline.pipeline_id
        )
        progress_fraction = (
            total_progress / total_weight if total_weight > 0 else 0.0
        )
        if not self._ui_enabled:
            return
        self._ui.update_warning_count(get_warning_count())
        self._ui.update_progress(
            pipeline.pipeline_id,
            progress_fraction,
            retries=pipeline.retries,
            failed=pipeline.failed,
        )
        self._ui.display()

    def _aggregate_pipeline_progress(
        self, pipeline_id: int
    ) -> tuple[float, float]:
        """
        Aggregate progress/weight recursively over a pipeline subtree.

        Retries and failed counters are intentionally not aggregated.
        """
        pipeline = self._pipelines[pipeline_id]
        total_progress = pipeline.weighted_progress
        total_weight = pipeline.total_weight

        for child_id in self._pipeline_children.get(pipeline_id, []):
            child_progress, child_weight = self._aggregate_pipeline_progress(
                child_id
            )
            total_progress += child_progress
            total_weight += child_weight

        return total_progress, total_weight

    def _refresh_pipeline_and_ancestors(self, pipeline_id: int) -> None:
        """Refresh one pipeline and all its ancestors for progress roll-up."""
        current = pipeline_id
        while current is not None:
            self._refresh_pipeline_bar(self._pipelines[current])
            current = self._pipeline_parent.get(current)

    def _set_running_state_for_lineage(self, pipeline_id: int) -> None:
        """Mark a pipeline and all its ancestors as running in the UI."""
        if not self._ui_enabled:
            return
        current = pipeline_id
        while current is not None:
            self._ui.update_state(current, "running")
            current = self._pipeline_parent.get(current)

    def _handle_started(
        self,
        task: TaskState,
        pipeline: PipelineState,
        total: int | None,
    ) -> None:
        if total is None:
            raise ValueError("total is required for started event")
        task.started_runs += 1

        # keep progress/total so retried tiles can rebuild rolled-back
        # progress.
        is_retry_pass = task.pending_retry_pass
        task.pending_retry_pass = False
        task.current_pass_is_retry = is_retry_pass
        if not is_retry_pass:
            # Use a nominal-only counter so retry passes do not incorrectly
            # trigger warnings about exceeding expected_runs
            task.nominal_started_runs += 1
            if task.nominal_started_runs > task.expected_runs:
                logging.warning(
                    f"ProgressTree warning: task {task.name} in pipeline "
                    f"{pipeline.name} started more times than expected "
                    f"({task.nominal_started_runs} > {task.expected_runs})"
                )
            if total <= 0:
                logging.warning(
                    f"ProgressTree warning: task {task.name} in "
                    f"pipeline {pipeline.name} started with "
                    f"non-positive total={total}"
                )
            task.total = total
            # Reset per-run counters so logging percentage stays within
            # 0-100 for each nominal run.
            task.progress_in_run = 0.0
            task.last_logged_percent = 0
            task.progressed_tiles = 0
            task.retries = 0
            task.failed = 0

        self._set_running_state_for_lineage(pipeline.pipeline_id)
        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
        add_progress_message(
            f"Started task {task.name} in "
            f"pipeline {pipeline.name}, total: {total}"
        )

    def _handle_progressed(
        self,
        task: TaskState,
        pipeline: PipelineState,
        value: int,
    ) -> None:
        if task.total <= 0:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} received 'progressed' before "
                f"a valid 'started' (total={task.total})"
            )
            return
        if value <= 0:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} received "
                f"non-positive progressed value={value}"
            )
            return
        run_share = task.weight / task.expected_runs
        increment = run_share * (float(value) / float(task.total))
        task.progress_in_run += increment
        pipeline.weighted_progress += increment
        task.progressed_tiles += int(value)

        # Log progress at 10% intervals
        current_count = max(0, task.progressed_tiles - task.retries)
        if current_count > task.total:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} effective progressed tiles "
                f"exceed total ({current_count} > {task.total})"
            )
        current_percent = (
            int((current_count / task.total) * 100) if task.total > 0 else 0
        )

        if current_percent >= task.last_logged_percent + 10:
            task.last_logged_percent = (current_percent // 10) * 10
            add_progress_message(
                f"Data list to process: {task.last_logged_percent}% "
                f"complete ({current_count}/{task.total} tiles) "
                f"[run {task.started_runs}/{task.expected_runs}]"
            )

        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)

    def _handle_retries(
        self,
        task: TaskState,
        pipeline: PipelineState,
        value: int,
    ) -> None:
        retries_count = int(value)
        if retries_count < 0:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} received "
                f"negative retries={retries_count}"
            )
            return
        task.retries += retries_count
        pipeline.retries += retries_count
        if retries_count > 0:
            task.pending_retry_pass = True
        if task.total > 0 and retries_count > task.total:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} received "
                f"retries greater than total ({retries_count} > {task.total})"
            )

        # Push progress back by the share represented by retried tiles.
        if task.total > 0:
            run_share = task.weight / task.expected_runs
            rollback = run_share * (float(retries_count) / float(task.total))
            rollback = min(rollback, task.progress_in_run)
            task.progress_in_run -= rollback
            pipeline.weighted_progress = max(
                0.0, pipeline.weighted_progress - rollback
            )

        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)

    def _handle_failed(
        self,
        task: TaskState,
        pipeline: PipelineState,
        value: int,
    ) -> None:
        failed_count = int(value)
        if failed_count < 0:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} received "
                f"negative failed={failed_count}"
            )
            return
        task.failed += failed_count
        pipeline.failed += failed_count
        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)

    def _handle_completed(
        self,
        task: TaskState,
        pipeline: PipelineState,
    ) -> None:
        if task.started_runs == 0:
            logging.warning(
                f"ProgressTree warning: task {task.name} in "
                f"pipeline {pipeline.name} completed without any start"
            )
        if not task.current_pass_is_retry:
            next_nominal_runs = task.completed_nominal_runs + 1
            if next_nominal_runs > task.expected_runs:
                logging.warning(
                    f"ProgressTree warning: task {task.name} in "
                    f"pipeline {pipeline.name} completed more times than "
                    f"expected ({next_nominal_runs} > {task.expected_runs})"
                )
            if task.started_runs < next_nominal_runs:
                logging.warning(
                    f"ProgressTree warning: task {task.name} in "
                    f"pipeline {pipeline.name} completed without matching "
                    f"starts ({task.started_runs} < {next_nominal_runs})"
                )
        if task.total > 0:
            run_share = task.weight / task.expected_runs
            if task.progress_in_run < run_share:
                missing = run_share - task.progress_in_run
                task.progress_in_run += missing
                pipeline.weighted_progress += missing

        # Count only nominal task runs toward expected_runs completion.
        if not task.current_pass_is_retry:
            task.completed_nominal_runs += 1
        task.current_pass_is_retry = False

        pipeline_task_ids = [
            tid
            for tid in self._tasks
            if self._task_to_pipeline[tid] == pipeline.pipeline_id
        ]
        all_tasks_done = all(
            self._tasks[tid].completed_nominal_runs
            >= self._tasks[tid].expected_runs
            for tid in pipeline_task_ids
        )
        total_expected = sum(
            self._tasks[tid].weight for tid in pipeline_task_ids
        )

        if all_tasks_done:
            pipeline.weighted_progress = total_expected
            if self._ui_enabled:
                self._ui.update_state(pipeline.pipeline_id, "completed")
        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)

    def notify(
        self,
        task_id: int,
        event: str,
        *,
        total: int | None = None,
        value: int = 1,
    ) -> None:
        """
        Handle task events from orchestrators and roll up to pipeline bars.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = self._tasks[task_id]
        pipeline = self._pipelines[task.pipeline_id]

        if event == "started":
            self._handle_started(task, pipeline, total)
            return

        if event == "progressed":
            self._handle_progressed(task, pipeline, value)
            return

        if event == "retries":
            self._handle_retries(task, pipeline, value)
            return

        if event == "failed":
            self._handle_failed(task, pipeline, value)
            return

        if event == "completed":
            self._handle_completed(task, pipeline)
            return

        raise ValueError(f"Unknown task event: {event}")

    def finalize(self) -> None:
        """Clean up UI resources (call after pipelines are done)."""
        # check task completion when the pipeline finished normally
        if getattr(self, "_finished_successfully", False):
            for task in self._tasks.values():
                pipeline = self._pipelines[task.pipeline_id]
                if task.completed_nominal_runs < task.expected_runs:
                    logging.warning(
                        f"ProgressTree warning: task {task.name} in "
                        f"pipeline {pipeline.name} finished with only "
                        f"{task.completed_nominal_runs}/{task.expected_runs} "
                        f"completed runs"
                    )

        if self._ui.live is not None:
            self._ui.live.stop()
            self._ui.live = None

    def draw(self) -> None:
        """Render the current tree state immediately."""
        if self._ui_enabled:
            self._ui.update_warning_count(get_warning_count())
            self._ui.display()

    def notify_crash(self, exception: BaseException) -> None:
        """Notify progress UI that a crash occurred and render details."""
        self._finished_successfully = False
        if self._ui_enabled:
            self._ui.update_warning_count(get_warning_count())
            self._ui.update_crash(exception)
            self._ui.display_final()

    def notify_success(self, output_dir: str | None) -> None:
        """Notify progress UI that pipeline completed successfully."""
        self._finished_successfully = True
        if self._ui_enabled:
            self._ui.update_warning_count(get_warning_count())
            self._ui.update_success(output_dir)
            self._ui.display_final()

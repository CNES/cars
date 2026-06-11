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

from .ui import PipelineTreeUI


@dataclass
class TaskState:
    """Internal state for one tracked task."""

    task_id: int
    name: str
    pipeline_id: int
    weight: float
    expected_runs: int = 1
    started_runs: int = 0
    progress_in_run: float = 0.0
    total: int = 0
    last_logged_percent: int = 0  # for logging progress at intervals


@dataclass
class PipelineState:
    """Internal state for one pipeline."""

    pipeline_id: int
    name: str
    position: int
    total_weight: float = 0.0
    weighted_progress: float = 0.0
    retries: int = 0


class ProgressTree:
    """Singleton registry with pipeline-level weighted progress bars."""

    _instance: ProgressTree | None = None

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
        self._SCALE = 1000
        self._ui = PipelineTreeUI(console=Console())
        self._pipeline_order: list[int] = []
        self._initialized = True

    def begin_pipeline(
        self, pipeline_name: str, parent_id: int | None = None
    ) -> int:
        """Declare that subsequent tasks belong to this pipeline."""
        pipeline_id = next(self._pipeline_id_gen)
        position = next(self._pipeline_position_gen)
        if parent_id is None:
            self._pipeline_order.append(pipeline_id)
        indent = 0 if parent_id is None else 1
        self._ui.add_node(
            pipeline_id, pipeline_name, indent=indent, parent_id=parent_id
        )
        self._ui.update_state(pipeline_id, "pending")
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
        self._tasks[task_id] = TaskState(
            task_id=task_id,
            name=task_name,
            pipeline_id=pipeline_id,
            weight=weight,
            expected_runs=expected_runs,
        )
        self._task_to_pipeline[task_id] = pipeline_id
        self._pipelines[pipeline_id].total_weight += weight

        logging.info(
            "Registered task '{}' under pipeline '{}' with weight {}".format(
                task_name, self._pipelines[pipeline_id].name, weight
            )
        )
        return task_id

    def _refresh_pipeline_bar(self, pipeline: PipelineState) -> None:
        """
        Update Rich UI display for one pipeline
        from weighted progress and retry metadata.
        """
        total_progress, total_weight, total_retries = (
            self._aggregate_pipeline_metrics(pipeline.pipeline_id)
        )
        progress_fraction = (
            total_progress / total_weight if total_weight > 0 else 0.0
        )
        pipeline.retries = total_retries
        self._ui.update_progress(
            pipeline.pipeline_id,
            progress_fraction,
            retries=pipeline.retries,
        )
        self._ui.display()

    def _aggregate_pipeline_metrics(
        self, pipeline_id: int
    ) -> tuple[float, float, int]:
        """
        Aggregate progress/weight/retries recursively over a pipeline subtree.
        """
        pipeline = self._pipelines[pipeline_id]
        total_progress = pipeline.weighted_progress
        total_weight = pipeline.total_weight
        total_retries = pipeline.retries

        for child_id in self._pipeline_children.get(pipeline_id, []):
            child_progress, child_weight, child_retries = (
                self._aggregate_pipeline_metrics(child_id)
            )
            total_progress += child_progress
            total_weight += child_weight
            total_retries += child_retries

        return total_progress, total_weight, total_retries

    def _refresh_pipeline_and_ancestors(self, pipeline_id: int) -> None:
        """Refresh one pipeline and all its ancestors."""
        current = pipeline_id
        while current is not None:
            self._refresh_pipeline_bar(self._pipelines[current])
            current = self._pipeline_parent.get(current)

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
            if total is None:
                raise ValueError("total is required for 'started' event")
            task.total = total
            task.started_runs += 1
            # Reset per-run counters so logging percentage stays within 0-100
            # for each new run of the same task.
            task.progress_in_run = 0.0
            task.last_logged_percent = 0
            if pipeline.weighted_progress == 0.0:
                self._ui.update_state(pipeline.pipeline_id, "running")
            self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
            logging.info(
                "Started task '{}' in pipeline '{}', total: {}".format(
                    task.name, pipeline.name, total
                )
            )
            return

        if event == "progressed":
            if task.total <= 0:
                return
            run_share = task.weight / task.expected_runs
            increment = run_share * (float(value) / float(task.total))
            task.progress_in_run += increment
            pipeline.weighted_progress += increment

            # Log progress at 10% intervals
            current_count = (
                int((task.progress_in_run / run_share) * task.total)
                if run_share > 0
                else 0
            )
            current_percent = (
                int((current_count / task.total) * 100) if task.total > 0 else 0
            )

            if current_percent >= task.last_logged_percent + 10:
                task.last_logged_percent = (current_percent // 10) * 10
                logging.info(
                    f"Data list to process: {task.last_logged_percent}% "
                    f"complete ({current_count}/{task.total} tiles) "
                    f"[run {task.started_runs}/{task.expected_runs}]"
                )

            self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
            return

        if event == "retries":
            pipeline.retries += int(value)
            self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
            return

        if event == "completed":
            if task.total > 0:
                run_share = task.weight / task.expected_runs
                if task.progress_in_run < run_share:
                    missing = run_share - task.progress_in_run
                    task.progress_in_run += missing
                    pipeline.weighted_progress += missing
            total_expected = sum(
                self._tasks[tid].weight
                for tid in self._tasks
                if self._task_to_pipeline[tid] == pipeline.pipeline_id
            )
            all_tasks_done = pipeline.weighted_progress >= total_expected - 1e-9
            if all_tasks_done:
                pipeline.weighted_progress = total_expected
                self._ui.update_state(pipeline.pipeline_id, "completed")
            self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
            return

        raise ValueError(f"Unknown task event: {event}")

    def finalize(self) -> None:
        """Clean up UI resources (call after pipelines are done)."""
        if self._ui.live is not None:
            self._ui.live.stop()

    def draw(self) -> None:
        """Render the current tree state immediately."""
        self._ui.display()

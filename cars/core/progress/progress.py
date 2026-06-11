"""Progress tree with Rich UI display.

One progress bar is shown per pipeline via Rich.
Tasks are registered under a pipeline with weights, and task
progress contributes to the pipeline bar accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from rich.console import Console

from cars.core.cars_logging import add_progress_message

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
    progressed_tiles: int = 0
    retries: int = 0
    failed: int = 0
    pending_retry_pass: bool = False


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
        self._ui_enabled = True
        self._pipeline_order: list[int] = []
        self._initialized = True

    def set_ui_enabled(self, enabled: bool) -> None:
        """Enable/disable Rich UI rendering for progress updates."""
        self._ui_enabled = bool(enabled)
        if not self._ui_enabled and self._ui.live is not None:
            self._ui.live.stop()
            self._ui.live = None

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

        add_progress_message(
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
        total_progress, total_weight = self._aggregate_pipeline_progress(
            pipeline.pipeline_id
        )
        progress_fraction = (
            total_progress / total_weight if total_weight > 0 else 0.0
        )
        if not self._ui_enabled:
            return
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

    def _handle_started(
        self,
        task: TaskState,
        pipeline: PipelineState,
        total: int | None,
    ) -> None:
        if total is None:
            raise ValueError("total is required for 'started' event")
        task.started_runs += 1

        # keep progress/total so retried tiles can rebuild rolled-back
        # progress.
        is_retry_pass = task.pending_retry_pass
        task.pending_retry_pass = False
        if not is_retry_pass:
            task.total = total
            # Reset per-run counters so logging percentage stays within
            # 0-100 for each nominal run.
            task.progress_in_run = 0.0
            task.last_logged_percent = 0
            task.progressed_tiles = 0
            task.retries = 0
            task.failed = 0

        if self._ui_enabled and pipeline.weighted_progress == 0.0:
            self._ui.update_state(pipeline.pipeline_id, "running")
        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)
        add_progress_message(
            "Started task '{}' in pipeline '{}', total: {}".format(
                task.name, pipeline.name, total
            )
        )

    def _handle_progressed(
        self,
        task: TaskState,
        pipeline: PipelineState,
        value: int,
    ) -> None:
        if task.total <= 0:
            return
        run_share = task.weight / task.expected_runs
        increment = run_share * (float(value) / float(task.total))
        task.progress_in_run += increment
        pipeline.weighted_progress += increment
        task.progressed_tiles += int(value)

        # Log progress at 10% intervals
        current_count = max(0, task.progressed_tiles - task.retries)
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
        task.retries += retries_count
        pipeline.retries += retries_count
        if retries_count > 0:
            task.pending_retry_pass = True

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
        task.failed += int(value)
        pipeline.failed += int(value)
        self._refresh_pipeline_and_ancestors(pipeline.pipeline_id)

    def _handle_completed(
        self,
        task: TaskState,
        pipeline: PipelineState,
    ) -> None:
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
        if self._ui.live is not None:
            self._ui.live.stop()

    def draw(self) -> None:
        """Render the current tree state immediately."""
        if self._ui_enabled:
            self._ui.display()

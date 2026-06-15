"""Rich-based UI for hierarchical pipeline progress display.

Displays a tree of pipelines with status icons, indentation, and progress bars.
Uses Rich's Live display to update content in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback
from rich.tree import Tree


@dataclass
class UINode:
    """Represents a displayable node in the pipeline tree."""

    name: str
    indent: int = 0
    pipeline_id: int | None = None
    progress: float = 0.0  # 0.0 to 1.0
    retries: int = 0
    failed: int = 0
    state: str = "pending"  # pending, running, completed
    children: list[UINode] = field(default_factory=list)

    def status_icon(self) -> str:
        """Return an icon for the current state."""
        if self.state == "completed":
            return "✓"
        if self.state == "running":
            return "⟳"
        # pending
        return "○"


class PipelineTreeUI:
    """Rich-based UI for displaying hierarchical pipeline progress."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.tree: dict[int, UINode] = {}
        self._order: list[int] = []
        self.live: Live | None = None
        self.warning_count: int = 0
        self.log_file_path: str | None = None
        self.crash_exception: BaseException | None = None
        self.success_output_dir: str | None = None

    def update_warning_count(self, warning_count: int) -> None:
        """Update total warning counter displayed in the panel footer."""
        self.warning_count = max(0, int(warning_count))

    def update_log_file_path(self, log_file_path: str | None) -> None:
        """Update log file path used by the footer log link button."""
        self.log_file_path = log_file_path

    def update_crash(self, exception: BaseException) -> None:
        """Store crash exception for rich crash rendering."""
        self.crash_exception = exception

    def update_success(self, output_dir: str | None) -> None:
        """Store successful output directory for final success rendering."""
        self.success_output_dir = output_dir

    def _path_to_uri(self, path_str: str) -> str:
        """Convert path string to clickable file URI."""
        try:
            return Path(path_str).resolve().as_uri()
        except ValueError:
            return "file://{}".format(quote(path_str, safe="/:"))

    def _render_log_button(self) -> Text:
        """Render a clickable footer button linking to the current log file."""
        if self.log_file_path:
            uri = self._path_to_uri(self.log_file_path)
            label = Text()
            label.append(
                "Open log file",
                style="grey50 underline link {}".format(uri),
            )
            # Visible URI fallback improves Ctrl+Click behavior in terminals
            # that do not fully support OSC 8 hyperlinks.
            # label.append(" (")
            # label.append(uri, style="underline")
            # label.append(")")
            return label
        return Text("", style="dim")

    def _render_success_panel(self) -> Panel | None:
        """Render final success panel with output links using Tree."""
        if not self.success_output_dir:
            return None

        output_path = Path(self.success_output_dir)
        output_uri = self._path_to_uri(self.success_output_dir)
        output_label = "📁 " + str(output_path) + "/"

        # Root tree node for output folder
        root = Tree(
            Text(
                output_label,
                style="underline link {}".format(output_uri),
            )
        )

        for entry_name in (
            "dsm",
            "depth_map",
            "point_cloud",
            "intermediate_data",
        ):
            entry_path = output_path / entry_name
            if entry_path.exists():
                entry_uri = self._path_to_uri(str(entry_path))
                entry_label = "📁 " + entry_name + "/"
                folder_node = root.add(
                    Text(
                        entry_label,
                        style="underline link {}".format(entry_uri),
                    )
                )

                # Show top-level files with links for main products.
                if entry_name in {"dsm", "depth_map", "point_cloud"}:
                    children = sorted(
                        entry_path.iterdir(),
                        key=lambda path_obj: path_obj.name.lower(),
                    )
                    for child_path in children:
                        child_uri = self._path_to_uri(str(child_path))
                        child_name = child_path.name
                        if child_path.is_dir():
                            child_label = "📁 " + child_name + "/"
                        else:
                            if child_path.suffix.lower() in {".tif", ".tiff"}:
                                child_label = "🖼  " + child_name
                            else:
                                child_label = "📄 " + child_name
                        folder_node.add(
                            Text(
                                child_label,
                                style="underline link {}".format(child_uri),
                            )
                        )
        # Put text above the tree to indicate success
        success_text = Text(
            "Output directory:",
            style="bold",
        )

        return Panel(
            Group(success_text, root),
            title="[bold green]Pipeline successful![/bold green]",
            border_style="green",
            expand=False,
        )

    def add_node(
        self,
        node_id: int,
        name: str,
        indent: int = 0,
        parent_id: int | None = None,
    ) -> None:
        """Add a node to the tree."""
        node = UINode(name=name, indent=indent, pipeline_id=node_id)
        self.tree[node_id] = node

        if parent_id is not None and parent_id in self.tree:
            # Add as child node, not to top-level order
            self.tree[parent_id].children.append(node)
        else:
            # Add top-level node to rendering order
            self._order.append(node_id)

    def update_state(self, node_id: int, state: str) -> None:
        """Update node state: pending, running, completed."""
        if node_id in self.tree:
            self.tree[node_id].state = state

    def update_progress(
        self,
        node_id: int,
        progress: float,
        retries: int = 0,
        failed: int = 0,
    ) -> None:
        """Update node progress (0.0 to 1.0) and retry count."""
        if node_id in self.tree:
            self.tree[node_id].progress = max(0.0, min(1.0, progress))
            self.tree[node_id].retries = retries
            self.tree[node_id].failed = failed

    def _node_label(self, node: UINode, depth: int) -> str:
        """Build a plain node label used for width calculation."""
        indent_str = "  " * depth
        display_name = node.name
        if re.match(r"^surface_modeling_res\d+$", node.name.lower()):
            display_name = "Surface Modeling"
        return f"{indent_str}{node.status_icon()} {display_name}"

    def _label_renderable(self, node: UINode, depth: int) -> RenderableType:
        """Render label with tabs first, then status icon/spinner and name."""
        display_name = node.name
        if re.match(r"^surface_modeling_res\d+$", node.name.lower()):
            display_name = "Surface Modeling"

        row = Table.grid(expand=False, padding=(0, 0))
        if depth > 0:
            row.add_column(no_wrap=True, width=depth * 2)  # for indentation
        row.add_column(no_wrap=True)  # for label and status

        if node.state == "running":
            # add spinner with label text after it
            status_and_name = Group(
                Spinner("dots", text=display_name, style="green"),
            )
        else:
            status_and_name = Text(f"{node.status_icon()} {display_name}")

        if depth > 0:
            row.add_row("", status_and_name)
        else:
            row.add_row(status_and_name)

        return row

    def _compute_label_width(self) -> int:
        """Compute a shared width so all bars start at the same column."""
        max_width = 0

        def walk(current: UINode, depth: int) -> None:
            nonlocal max_width
            max_width = max(max_width, len(self._node_label(current, depth)))
            for child in current.children:
                walk(child, depth + 1)

        for node_id in self._order:
            walk(self.tree[node_id], 0)

        return max(max_width, 24)

    def _render_node(
        self,
        node: UINode,
        depth: int = 0,
        label_width: int = 24,
    ) -> list[RenderableType]:
        """
        Render a single node and its children as a list of Rich renderables.
        """
        lines: list[RenderableType] = []
        label = self._label_renderable(node, depth)

        # Build a compact left-aligned row: name | bar | percent/retries
        row = Table.grid(expand=False, padding=(0, 0))
        row.add_column(no_wrap=True, width=label_width)
        row.add_column(no_wrap=True)
        row.add_column(no_wrap=True)

        progress = max(0.0, min(1.0, node.progress))
        progress_bar = ProgressBar(total=1.0, completed=progress, width=20)
        suffix_parts = []
        if node.retries > 0:
            suffix_parts.append(f"retries={node.retries}")
        if node.failed > 0:
            suffix_parts.append(f"failed={node.failed}")
        status_suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
        progress_label = Text(
            f" {progress * 100:.0f}%{status_suffix}", no_wrap=True
        )

        row.add_row(label, progress_bar, progress_label)
        lines.append(row)

        # Recursively render children with increased depth
        for child in node.children:
            child_lines = self._render_node(
                child,
                depth=depth + 1,
                label_width=label_width,
            )
            lines.extend(child_lines)

        return lines

    def _render_full_tree(self) -> Panel:
        """Render the full tree as a Rich Panel."""
        lines: list[RenderableType] = []
        label_width = self._compute_label_width()

        for node_id in self._order:
            node = self.tree[node_id]
            base_depth = 0

            # Check if this node is a resolution marker
            if "res" in node.name.lower():
                # Extract resolution number if present
                match = re.search(r"res(\d+)", node.name.lower())
                if match:
                    res_num = match.group(1)
                    section_text = Text(f"Resolution {res_num}:", style="bold")
                    lines.append(section_text)
                    base_depth = 1

            # Render node and its children
            node_lines = self._render_node(
                node, depth=base_depth, label_width=label_width
            )
            lines.extend(node_lines)

        # Main tree section
        if not lines:
            content: RenderableType = Text("No pipelines yet")
        else:
            footer_row = Table.grid(expand=True)
            footer_row.add_column(justify="left")
            footer_row.add_column(justify="right")
            footer_row.add_row(
                Text(f"Warnings: {self.warning_count}"),
                self._render_log_button(),
            )
            content = Group(*lines, Rule(style="grey50"), footer_row)

        main_panel = Panel(
            content, title="[bold]CARS Progress[/bold]", expand=False
        )

        # Crash section (if any)
        if self.crash_exception is not None:
            crash_summary = Text(
                "{}: {}".format(
                    type(self.crash_exception).__name__,
                    self.crash_exception,
                ),
                style="bold red",
            )
            crash_traceback = Traceback.from_exception(
                type(self.crash_exception),
                self.crash_exception,
                self.crash_exception.__traceback__,
                show_locals=False,
            )
            crash_panel = Panel(
                Group(crash_summary, crash_traceback),
                title="[bold red]Crash[/bold red]",
                border_style="red",
                expand=False,
            )
            return Group(main_panel, Text(), crash_panel)

        success_panel = self._render_success_panel()
        if success_panel is not None:
            return Group(main_panel, Text(), success_panel)

        return main_panel

    def display(self) -> None:
        """Render the full tree to console using Live display."""
        panel = self._render_full_tree()

        if self.live is None:
            self.live = Live(panel, console=self.console, refresh_per_second=10)
            self.live.start()
        else:
            self.live.update(panel)

    def display_final(self) -> None:
        """Stop Live display and display the final rendered view once."""
        panel = self._render_full_tree()
        if self.live is not None:
            # update one last time to show final state before stopping live
            self.live.update(panel)
            self.live.stop()
            self.live = None
        else:
            # print the panel if live never started
            self.console.print(panel)

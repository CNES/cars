"""Rich-based UI for hierarchical pipeline progress display.

Displays a tree of pipelines with status icons, indentation, and progress bars.
Uses Rich's Live display to update content in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


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
        elif self.state == "running":
            return "⟳"
        else:  # pending
            return "○"


class PipelineTreeUI:
    """Rich-based UI for displaying hierarchical pipeline progress."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.tree: dict[int, UINode] = {}
        self._order: list[int] = []
        self.live: Live | None = None

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
        """Build a node label including indentation and status icon."""
        indent_str = "  " * depth
        icon = node.status_icon()
        display_name = node.name
        if re.match(r"^surface_modeling_res\d+$", node.name.lower()):
            display_name = "Surface Modeling"
        return f"{indent_str}{icon} {display_name}"

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
        line = self._node_label(node, depth)

        # Build a compact left-aligned row: name | bar | percent/retries
        row = Table.grid(expand=False, padding=(0, 1))
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
            f"{progress * 100:.0f}%{status_suffix}", no_wrap=True
        )

        row.add_row(Text(line), progress_bar, progress_label)
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

        # Combine all lines into a panel
        if not lines:
            content = Text("No pipelines yet")
        else:
            content = Group(*lines)

        return Panel(content, title="[bold]CARS Progress[/bold]", expand=False)

    def display(self) -> None:
        """Render the full tree to console using Live display."""
        panel = self._render_full_tree()

        if self.live is None:
            self.live = Live(panel, console=self.console, refresh_per_second=10)
            self.live.start()
        else:
            self.live.update(panel)

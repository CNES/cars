# pylint: skip-file
# flake8: noqa

import html
import json
from pathlib import Path

import yaml
from docutils import nodes
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from sphinx.util.docutils import SphinxDirective


def convert_yaml_to_json(app, conf):
    """
    On build start: ensure every *.yaml file has a matching *.json file.
    """
    src_dir = Path(app.srcdir) / "example_configs"

    for yaml_file in src_dir.rglob("*.yaml"):
        json_file = yaml_file.with_suffix(".json")
        with open(yaml_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


class IncludeCarsConfigDirective(SphinxDirective):
    """
    Directive: .. include-cars-config:: path/to/myfile
    """

    required_arguments = 1

    def run(self):
        base = self.arguments[0]

        yaml_code = self._read_file(base + ".yaml", "yaml")
        json_code = self._read_file(base + ".json", "json")

        html = f"""
            <div class="cars-tabs">
            <div class="cars-tabs-buttons">
                <button class="cars-tab-btn active" onclick="carsSwitchTab(this, 'yaml')">YAML</button>
                <button class="cars-tab-btn" onclick="carsSwitchTab(this, 'json')">JSON</button>
                <button class="cars-tab-btn cars-copy-btn" onclick="carsCopyCode(this)">Copy</button>
            </div>

            <div class="cars-tab-content yaml active">
                <pre class="highlight">{yaml_code}</pre>
            </div>

            <div class="cars-tab-content json">
                <pre class="highlight">{json_code}</pre>
            </div>
            </div>
        """

        raw_node = nodes.raw("", html, format="html")
        return [raw_node]

    def _read_file(self, filename: str, language: str) -> str:
        """Read file contents safely for embedding."""

        src_path = Path(self.state.document.current_source).parent / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Config file not found: {src_path}")
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()

        lexer = get_lexer_by_name(language, stripall=False)
        formatter = HtmlFormatter(nowrap=True)
        return highlight(content, lexer, formatter)


def setup(app):
    app.connect("config-inited", convert_yaml_to_json)
    app.add_directive("include-cars-config", IncludeCarsConfigDirective)

    # Attach custom JS + CSS
    app.add_css_file("css/cars_tabs.css")
    app.add_js_file("js/cars_tabs.js")

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

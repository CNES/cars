# pylint: skip-file
# flake8: noqa

import html
import io
import json
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
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


class IncludeCarsConfigDirectiveSection(SphinxDirective):
    """
    Directive: .. include-cars-config-section:: path/to/myfile
    """

    required_arguments = 1
    option_spec = {
        "key": directives.unchanged,  # ex: "applications:1"
    }

    def run(self):
        base = self.arguments[0]
        key_option = self.options.get("key")

        src_path_yaml = Path(self.state.document.current_source).parent / (
            base + ".yaml"
        )
        src_path_json = Path(self.state.document.current_source).parent / (
            base + ".json"
        )

        # --- Lire YAML principal ---
        if not src_path_yaml.exists():
            raise FileNotFoundError(f"YAML file not found: {src_path_yaml}")
        with open(src_path_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # --- Section à afficher ---
        section_data = data
        if key_option:
            keys = [k.strip() for k in key_option.split(":")]
            parent_key = str(keys[0])
            child_key = str(keys[1]) if len(keys) > 1 else None

            if parent_key not in data:
                raise ValueError(f"Parent key '{parent_key}' not found in YAML")
            parent_data = data[parent_key]

            if child_key:
                if child_key not in parent_data:
                    raise ValueError(
                        f"Child key '{child_key}' not found in '{parent_key}'"
                    )
                # Cas spécial applications:1
                if child_key == "1":
                    section_data = {
                        parent_key: {child_key: parent_data[child_key]}
                    }
                else:
                    section_data = {child_key: parent_data[child_key]}
            else:
                section_data = {parent_key: parent_data}

        # --- YAML / JSON pour la section ---
        yaml_buf = io.StringIO()
        yaml.dump(section_data, yaml_buf, sort_keys=False)
        yaml_text = yaml_buf.getvalue()
        json_text = json.dumps(section_data, indent=4)

        yaml_highlighted = highlight(
            yaml_text,
            get_lexer_by_name("yaml"),
            HtmlFormatter(nowrap=True, noclasses=True, style="default"),
        )
        json_highlighted = highlight(
            json_text,
            get_lexer_by_name("json"),
            HtmlFormatter(nowrap=True, noclasses=True, style="default"),
        )

        # --- Charger YAML complet pour bouton copy full ---
        with open(src_path_yaml, "r", encoding="utf-8") as f:
            full_yaml = f.read()
        escaped_full_yaml = full_yaml.replace("`", "\\`").replace("$", "\\$")

        # --- Charger JSON complet pour bouton copy full ---
        with open(src_path_json, "r", encoding="utf-8") as f:
            full_json = f.read()
        escaped_full_json = html.escape(full_json)

        # --- HTML avec boutons dans la même barre ---
        html_content = f"""
        <div class="cars-tabs"  data-full-yaml="{escaped_full_yaml}" data-full-json="{escaped_full_json}">
            <div class="cars-tabs-buttons">
                <button class="cars-tab-btn active" onclick="carsSwitchTab(this, 'yaml')">YAML</button>
                <button class="cars-tab-btn" onclick="carsSwitchTab(this, 'json')">JSON</button>
                <div class="cars-buttons-spacer"></div>
                <button class="cars-tab-btn cars-copy-btn" onclick="carsCopyCode(this)">Copy section</button>
                <button class="cars-tab-btn cars-copy-btn"
                        onclick="carsCopyFull(this, `{escaped_full_yaml}`)">
                    Copy full config
                </button>
            </div>

            <div class="cars-tab-content yaml active">
                <div class="cars-config-scrollable">
                    <pre class="highlight">{yaml_highlighted}</pre>
                </div>
            </div>

            <div class="cars-tab-content json">
                <div class="cars-config-scrollable">
                    <pre class="highlight">{json_highlighted}</pre>
                </div>
            </div>
        </div>
        """

        return [nodes.raw("", html_content, format="html")]


class IncludeCarsConfigDirective(SphinxDirective):
    """
    Directive: .. include-cars-config:: path/to/myfile
    """

    required_arguments = 1
    option_spec = {
        "json": directives.unchanged,
    }

    def run(self):
        base = self.arguments[0]
        json_option = True
        if "json" in self.options:
            val = str(self.options["json"]).strip().lower()
            if val == "false":
                json_option = False

        yaml_code = self._read_file(base + ".yaml", "yaml")

        json_code = ""
        json_button = ""
        if json_option:
            json_code = self._read_file(base + ".json", "json")
            json_button = '<button class="cars-tab-btn" onclick="carsSwitchTab(this, \'json\')">JSON</button>'

        html_content = f"""
            <div class="cars-tabs">
            <div class="cars-tabs-buttons">
                <button class="cars-tab-btn active" onclick="carsSwitchTab(this, 'yaml')">YAML</button>
                {json_button}
                <button class="cars-tab-btn cars-copy-btn" onclick="carsCopyCode(this)">Copy</button>
            </div>

            <div class="cars-tab-content yaml active">
                <pre class="highlight">{yaml_code}</pre>
            </div>

            {(
                f'''
                <div class="cars-tab-content json">
                    <pre class="highlight">{json_code}</pre>
                </div>
                ''' if json_option else ''
            )}
            </div>
        """

        raw_node = nodes.raw("", html_content, format="html")
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
    app.add_directive(
        "include-cars-config-section", IncludeCarsConfigDirectiveSection
    )

    # Attach custom JS + CSS
    app.add_css_file("css/cars_tabs.css")
    app.add_js_file("js/cars_tabs.js")

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

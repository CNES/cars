# pylint: skip-file
# flake8: noqa
# This file contains the CARS extension for Sphinx

import json
from pathlib import Path

import yaml
from docutils import nodes
from docutils.statemachine import StringList
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
    (without extension, relative to source directory)

    Example:
        .. include-cars-config:: example_configs/config
    """

    required_arguments = 1

    def run(self):
        base = self.arguments[0]

        # Build the reST we want to include
        rst = f"""
.. tabs::

   .. tab:: YAML
      .. literalinclude:: {base}.yaml
         :language: yaml

   .. tab:: JSON
      .. literalinclude:: {base}.json
         :language: json
"""

        # Convert to StringList for nested_parse
        rst_lines = StringList(rst.splitlines())

        section = nodes.section()
        section.document = self.state.document
        self.state.nested_parse(rst_lines, 0, section)

        return section.children


def setup(app):
    app.connect("config-inited", convert_yaml_to_json)
    app.add_directive("include-cars-config", IncludeCarsConfigDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

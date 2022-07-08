CARS Documentation generation
=============================

CARS documentation is based on (Sphinx)[https://www.sphinx-doc.org/] and is in `docs/source` directory.

Use the following command line at CARS source code root directly

```
make docs
```

for automated CARS installation and documentation generation in `docs/build`

Otherwise follow the steps below:


CARS installation with Sphinx dependencies
------------------------------------------

First, create a virtualenv and install CARS  following [CARS Installation](./docs/source/install.rst)

You can use the following command line at CARS root directory:

```
make install
```

This install CARS in a virtualenv with sphinx documentation dependencies using : `pip install .[doc]`  

The autodoc needs indeed CARS installed for modules API.


CARS Sphinx documentation
-------------------------

Go to `docs` directory from CARS source root directory.

```
cd docs
```

For HTML documentation generation:
```
make html
```

For PDF generation :
```
make latexpdf
```

To clean generated documentation :
```
make clean
```

Plantuml diagrams
-----------------

To generate/update CARS [plantUML](https://plantuml.com/) documentation diagrams:
* Install plantuml on your system.
* Go to diagram directory
* Use following command line to generate in SVG:
```
plantuml -tsvg diagram.pu
```

Diagrams source and associated SVG are in `docs/source/diagrams/` directory

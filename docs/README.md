CARS Documentation generation
=============================

CARS documentation is based on (Sphinx)[https://www.sphinx-doc.org/] and is in `docs/source` directory.

Use the following command line at CARS source code root directly

```
make doc
```

for automated CARS installation and documentation generation in `docs/build`

Otherwise follow the steps below:


CARS installation with Sphinx dependencies
------------------------------------------

First, create a virtualenv and install CARS  following [CARS Installation](./docs/source/install.rst)

You can use the following command line at CARS root directory:

```
make install-doc
```

This install CARS in a virtualenv with sphinx documentation dependencies using : `pip install .[doc]`  

The autodoc needs indeed CARS installed for modules API.


CARS Sphinx documentation
-------------------------

Go to `docs` directory from CARS source root directory.

```
cd docs
```

First generate Autodoc generation in docs directory:
````
sphinx-apidoc -o source/apidoc/ ../cars
````

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

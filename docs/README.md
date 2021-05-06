CARS Documentation generation
=============================

First, create a virtualenv and install CARS  following [CARS Installation](./docs/source/install.rst)

The autodoc needs indeed CARS installed for modules API.

Then, go to `docs` directory.

Install CARS doc added requirements in requirements-doc.txt
```
    $ pip install -r requirements-doc.txt
```

For Autodoc generation in docs directory:
````
sphinx-apidoc -o source/apidoc/ ../cars
````

For HTML documentation generation:
```
make html
```

For PDF generation :
```
sphinx-build -b rinoh source _build/rinoh
```

To clean generated documentation :
```
make clean
```

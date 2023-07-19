=================
How to contribute
=================

CARS is an open source software : don't hesitate to hack it and contribute !

Please go to `the GitHub repository`_  for source code.

Read also `CARS Contribution guide`_ with `LICENCE <https://raw.githubusercontent.com/CNES/cars/master/LICENSE>`_ and `Contributor Licence Agrements <https://github.com/CNES/cars/tree/master/docs/source/CLA>`_.

**Contact:** cars AT cnes.fr

Developer Install
=================
:ref:`Install` procedure is globally followed but adapted to get CARS development environment.
Obviously, we recommend to use a `virtualenv`_ environment, so that :term:`CARS` do not interfere with other packages installed on your
system.

* Install `OTB`_ and `VLFeat`_: see :ref:`dependencies`.

* Clone CARS repository from GitHub :

.. code-block:: console

    $ git clone https://github.com/CNES/cars.git
    $ cd cars

* Install CARS in a `virtualenv`_ in developer mode

.. code-block:: console

    $ make install-dev # CARS installed in ``venv`` virtualenv

* Run CARS in `virtualenv`_

.. code-block:: console

    $ source venv/bin/activate
    $ source venv/bin/env_cars.sh
    $ cars -h

The detailed development install method is described in `Makefile <https://raw.githubusercontent.com/CNES/cars/master/Makefile>`_

Particularly, it uses the following pip editable install:

.. code-block:: console

    pip install -e .[dev]

With this pip install mode, source code modifications directly impacts ``cars`` command line.

Coding guide
============

Here are some rules to apply when developing a new functionality:

* **Comments:** Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* **Test**: Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* **Documentation**: All functions shall be documented (object, parameters, return values).
* **Use type hints**: Use the type hints provided by the `typing` python module.
* **Use doctype**: Follow sphinx default doctype for automatic API
* **Quality code**: Correct project quality code errors with pre-commit automatic workflow (see below)
* **Factorization**: Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the cars python modules.
* **Be careful with user interface upgrade:** If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* **Logging and no print**: The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* **Limit classes**: If possible, limit the use of classes at one or 2 levels and opt for a functional approach when possible. The classes are reserved for data modelling if it is impossible to do so using `xarray` and for the good level of modularity.
* **Limit new dependencies**: Do not add new dependencies unless it is absolutely necessary, and only if it has a **permissive license**.

Pre-commit validation
=====================

A pre-commit validation is installed with code quality tools (see below).
It is installed automatically by `make install-dev` command.

Here is the way to install it manually:

.. code-block:: console

  $ pre-commit install -t pre-commit # for commit rules
  $ pre-commit install -t pre-push   # for push rules

This installs the pre-commit hook in `.git/hooks/pre-commit` and `.git/hooks/pre-push`  from `.pre-commit-config.yaml <https://raw.githubusercontent.com/CNES/cars/master/.pre-commit-config.yaml>`_ file configuration.

It is possible to test pre-commit before committing:

.. code-block:: console

  $ pre-commit run --all-files                # Run all hooks on all files
  $ pre-commit run --files cars/__init__.py   # Run all hooks on one file
  $ pre-commit run pylint                     # Run only pylint hook


Documentation
=============

CARS contains its Sphinx Documentation in the code in docs directory.

To generate documentation, use:

.. code-block:: console

  $ make docs
  
The documentation is then build in docs/build directory and can be consulted with a web browser.

Documentation can be edited in docs/source/ RST files.

Jupyter notebooks tutorials
============================

CARS contains notebooks and quick start scripts in tutorials directory.

To generate a Jupyter kernel with CARS installation, use:

.. code-block:: console

  $ make notebook
  
Follow indications to run a jupyter notebook.

Kernel is created with following command (with cars-version updated):

.. code-block:: console

  $ python -m ipykernel install --sys-prefix --name=cars-version --display-name=cars-version

To run the jupyter notebook, use:

.. code-block:: console

  $ jupyter notebook


Code quality
=============
CARS uses `Isort`_, `Black`_, `Flake8`_ and `Pylint`_ quality code checking.

Use the following command in CARS `virtualenv`_ to check the code with these tools:

.. code-block:: console

    $ make lint

Use the following command to format the code with isort and black:

.. code-block:: console

    $ make format

Isort
-----
`Isort`_ is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

CARS ``isort`` configuration is done in `pyproject.toml`_

`Isort`_ manual usage examples:

.. code-block:: console

    $ cd CARS_HOME
    $ isort --check cars tests  # Check code with isort, does nothing
    $ isort --diff cars tests   # Show isort diff modifications
    $ isort cars tests          # Apply modifications

`Isort`_ messages can be avoided when really needed with **"# isort:skip"** on the incriminated line.

Black
-----
`Black`_ is a quick and deterministic code formatter to help focus on the content.

CARS ``black`` configuration is done in `pyproject.toml`_

If necessary, Black doesn’t reformat blocks that start with "# fmt: off" and end with # fmt: on, or lines that ends with "# fmt: skip". "# fmt: on/off" have to be on the same level of indentation.

`Black`_ manual usage examples:

.. code-block:: console

    $ cd CARS_HOME
    $ black --check cars tests  # Check code with black with no modifications
    $ black --diff cars tests   # Show black diff modifications
    $ black cars tests          # Apply modifications

Flake8
------
`Flake8`_ is a command-line utility for enforcing style consistency across Python projects. By default it includes lint checks provided by the PyFlakes project, PEP-0008 inspired style checks provided by the PyCodeStyle project, and McCabe complexity checking provided by the McCabe project. It will also run third-party extensions if they are found and installed.

CARS ``flake8`` configuration is done in `setup.cfg <https://raw.githubusercontent.com/CNES/cars/master/setup.cfg>`_

`Flake8`_ messages can be avoided (in particular cases !) adding "# noqa" in the file or line for all messages.
It is better to choose filter message with "# noqa: E731" (with E371 example being the error number).
Look at examples in source code.

Flake8 manual usage examples:

.. code-block:: console

  $ cd CARS_HOME
  $ flake8 cars tests           # Run all flake8 tests


Pylint
------
`Pylint`_ is a global linting tool which helps to have many information on source code.

CARS ``pylint`` configuration is done in dedicated `.pylintrc <//https://raw.githubusercontent.com/CNES/cars/master/.pylintrc>`_ file.

`Pylint`_ messages can be avoided (in particular cases !) adding "# pylint: disable=error-message-name" in the file or line.
Look at examples in source code.

Pylint manual usage examples:

.. code-block:: console

  $ cd CARS_HOME
  $ pylint tests cars       # Run all pylint tests
  $ pylint --list-msgs          # Get pylint detailed errors information


Tests
======

CARS includes a set of tests executed with `pytest <https://docs.pytest.org/>`_ tool.

To launch tests:

.. code-block:: console

    make test

It launches only the ``unit_tests`` and ``pbs_cluster_tests`` test targets

Before the tests execution, the ``CARS_TEST_TEMPORARY_DIR`` can be defined to indicate where to write the temporary data bound to the test procedure (if the variable is not set, cars will use current working directory).

Several kinds of tests are identified by specific pytest markers:

- the unit tests defined by the ``unit_tests`` marker: ``make test-unit``
- the PBS cluster tests defined by the ``pbs_cluster_tests`` marker: ``make test-pbs-cluster``
- the SLURM cluster tests defined by the ``slurm_cluster_tests`` marker: ``make test-slurm-cluster``
- the Jupyter notebooks test defined by the ``notebook_tests`` marker: ``make test-notebook``

Advanced testing
----------------

To execute the tests manually, use ``pytest`` at the CARS projects's root (after initializing the environment):

.. code-block:: console

    $ python -m pytest

To run only the unit tests:

.. code-block:: console

    $ cd cars/
    $ pytest -m unit_tests

To run only the PBS cluster tests:

.. code-block:: console

    $ cd cars/
    $ pytest -m pbs_cluster_tests

To run only the Jupyter notebooks tests:

.. code-block:: console

    $ cd cars/
    $ pytest -m notebook_tests

It is possible to obtain the code coverage level of the tests by installing the ``pytest-cov`` module and use the ``--cov`` option.

.. code-block:: console

    $ cd cars/
    $ python -m pytest --cov=cars

It is also possible to execute only a specific part of the test, either by indicating the test file to run:

.. code-block:: console

    $ cd cars/
    $ python -m pytest tests/test_tiling.py

Or by using the ``-k`` option which will execute the tests which names contain the option's value:

.. code-block:: console

    $ cd cars/
    $ python -m pytest -k end2end

By default, ``pytest`` does not display the traces generated by the tests but only the tests' status (passed or failed). To get all traces, the following options have to be added to the command line (which can be combined with the previous options):

.. code-block:: console

    $ cd cars/
    $ python -m pytest -s -o log_cli=true -o log_cli_level=INFO


.. _`OTB`: https://www.orfeo-toolbox.org/CookBook/Installation.html
.. _`VLFeat`: https://www.vlfeat.org/compiling-unix.html
.. _`the GitHub repository`: https://github.com/CNES/cars
.. _`CARS Contribution guide`: https://github.com/CNES/cars/blob/master/CONTRIBUTING.md
.. _`virtualenv`: https://virtualenv.pypa.io/
.. _`Isort`: https://pycqa.github.io/isort/
.. _`Black`: https://black.readthedocs.io/
.. _`Flake8`: https://flake8.pycqa.org/
.. _`Pylint`: http://pylint.pycqa.org/
.. _`pyproject.toml`: https://raw.githubusercontent.com/CNES/cars/master/pyproject.toml

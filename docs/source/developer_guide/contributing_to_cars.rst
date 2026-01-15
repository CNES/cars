
.. _contributing_to_cars:

====================
Contributing to CARS
====================

CARS is an open source software : don't hesitate to hack it and contribute !

Please go to `the GitHub repository`_  for source code.

Read also `CARS Contribution guide`_ with `LICENCE <https://raw.githubusercontent.com/CNES/cars/master/LICENSE>`_ and `Contributor Licence Agrements <https://github.com/CNES/cars/tree/master/docs/source/CLA>`_.

**Contact:** cars AT cnes.fr



Proposing new features
======================

To propose a new feature, start by opening an issue on GitHub to describe your idea.
Discuss the feature with the core developers either through the issue or on the dedicated Slack channel.
Once aligned, clone the repository, set it up with pre-commit hooks, and ensure your implementation adheres to the project’s documentation, testing, and coding guidelines.
Finally, submit a pull request (PR) for review by the core team.

Build CARS for developers
=========================

We recommend to use a `virtualenv`_ environment, so that :term:`CARS` do not interfere with other packages installed on your system.

* Clone CARS repository from GitHub :

.. code-block:: console

    git clone https://github.com/CNES/cars.git
    cd cars

* Install CARS in a `virtualenv`_ in developer mode

.. code-block:: console

    make install/dev # CARS installed in ``venv`` virtualenv

* Run CARS in `virtualenv`_

.. code-block:: console

    source venv/bin/activate
    source venv/bin/env_cars.sh
    cars -h

The detailed development install method is described in `Makefile <https://raw.githubusercontent.com/CNES/cars/master/Makefile>`_

Particularly, it uses the following pip editable install:

.. code-block:: console

    pip install -e .[dev]

With this pip install mode, source code modifications directly impacts ``cars`` command line.


If your machine already has local installations of **GDAL**, **Fiona**, or **PROJ**, you must install them in **``--no-binary``** mode before installing this tool. This ensures that locally compiled versions are used, preventing conflicts with precompiled binaries.

One straightforward solution is to use the provided script in the repository:

.. code-block:: bash

   make install/dev-gdal

This command installs the required dependencies by compiling the packages from their source, ensuring optimal compatibility with your environment.


Setting up a development environment with docker
================================================

To setup a development environment with docker, run the following command:

.. code-block:: console

    docker build -t cars-dev -f Dockerfile .
    docker run -it  -v "$(pwd)":/app/cars  --entrypoint=/bin/bash cars-dev

You're ready to use CARS, all files in the current directory are mounted in the container.


Code Guideline
==============

Reusing CARS Concepts
---------------------

When contributing to this project, ensure that your implementation aligns with the **concepts and patterns** already established in **CARS**. For details, refer to the `Concepts` section of the documentation.

Adding New Libraries
--------------------

Before introducing new libraries, verify that their **license is compatible** with the project. For a list of allowed licenses, see the `Licensing` section.

Adding C++ Code
---------------

C++ code should be integrated as **plugins** to maintain modularity and avoid bloating the core codebase. Use **pybind11** to create Python wrappers for C++ functionality. This ensures seamless integration with the Python interface while keeping the C++ logic encapsulated.

For examples and best practices, refer to the existing bindings in the project:
* resampling application -> cars-resample
* rasterization application -> cars-rasterize


Documentation Guideline
=======================


CARS contains its Sphinx Documentation in the code in docs directory.

To generate documentation, use:

.. code-block:: console

  make docs

The documentation is then build in docs/build directory and can be consulted with a web browser.

Documentation can be edited in docs/source/ RST files.


Documentation compilation
-------------------------

The documentation is automatically compiled pre-push, meaning it is built and validated every time you push changes to the Git repository.
To ensure a smooth process and avoid compilation errors, it is strongly recommended to have CARS installed with pre-commit hooks.
This setup allows you to verify locally that the documentation compiles correctly before pushing your changes.

Tests Guideline
===============

CARS includes a set of tests executed with `pytest <https://docs.pytest.org/>`_ tool.

To launch tests:

.. code-block:: console

    make test

It launches only the ``unit_tests`` and ``pbs_cluster_tests`` test targets

Before the tests execution, the ``CARS_TEST_TEMPORARY_DIR`` can be defined to indicate where to write the temporary data bound to the test procedure (if the variable is not set, cars will use default temporary directory).

Several kinds of tests are identified by specific pytest markers:

- the unit tests defined by the ``unit_tests`` marker: ``make test-unit``
- the PBS cluster tests defined by the ``pbs_cluster_tests`` marker: ``make test-pbs-cluster``
- the SLURM cluster tests defined by the ``slurm_cluster_tests`` marker: ``make test-slurm-cluster``
- the Jupyter notebooks test defined by the ``notebook_tests`` marker: ``make test-notebook``

Advanced testing
----------------

To execute the tests manually, use ``pytest`` at the CARS projects's root (after initializing the environment):

.. code-block:: console

    python -m pytest

To run only the unit tests:

.. code-block:: console

    cd cars/
    pytest -m unit_tests

To run only the PBS cluster tests:

.. code-block:: console

    cd cars/
    pytest -m pbs_cluster_tests

To run only the Jupyter notebooks tests:

.. code-block:: console

    cd cars/
    pytest -m notebook_tests

It is possible to obtain the code coverage level of the tests by installing the ``pytest-cov`` module and use the ``--cov`` option.

.. code-block:: console

    cd cars/
    python -m pytest --cov=cars

It is also possible to execute only a specific part of the test, either by indicating the test file to run:

.. code-block:: console

    cd cars/
    python -m pytest tests/test_tiling.py

Or by using the ``-k`` option which will execute the tests which names contain the option's value:

.. code-block:: console

    cd cars/
    python -m pytest -k end2end

By default, ``pytest`` does not display the traces generated by the tests but only the tests' status (passed or failed). To get all traces, the following options have to be added to the command line (which can be combined with the previous options):

.. code-block:: console

    cd cars/
    python -m pytest -s -o log_cli=true -o log_cli_level=INFO


Stylistic Guideline, Code Quality
=================================

Here are some rules to apply when developing a new functionality:

* **Comments:** Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* **Test**: Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* **Documentation**: All functions shall be documented (object, parameters, return values).
* **Use type hints**: Use the type hints provided by the `typing` python module.
* **Use doctype**: Follow sphinx default doctype for automatic API.
* **Quality code**: Correct project quality code errors with pre-commit automatic workflow (see below).
* **Factorization**: Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the cars python modules.
* **Be careful with user interface upgrade:** If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* **Logging and no print**: The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* **Limit classes**: If possible, limit the use of classes at one or 2 levels and opt for a functional approach when possible. The classes are reserved for data modelling if it is impossible to do so using `xarray` and for the good level of modularity.
* **Limit new dependencies**: Do not add new dependencies unless it is absolutely necessary, and only if it has a **permissive license**.

Pre-commit validation
---------------------

A pre-commit validation is installed with code quality tools (see below).
It is installed automatically by `make install-dev` command.

Here is the way to install it manually:

.. code-block:: console

  pre-commit install -t pre-commit # for commit rules
  pre-commit install -t pre-push   # for push rules

This installs the pre-commit hook in `.git/hooks/pre-commit` and `.git/hooks/pre-push`  from `.pre-commit-config.yaml <https://raw.githubusercontent.com/CNES/cars/master/.pre-commit-config.yaml>`_ file configuration.

It is possible to test pre-commit before committing:

.. code-block:: console

  pre-commit run --all-files                # Run all hooks on all files
  pre-commit run --files cars/__init__.py   # Run all hooks on one file
  pre-commit run pylint                     # Run only pylint hook



CARS uses `Isort`_, `Black`_, `Flake8`_ and `Pylint`_ quality code checking.

Use the following command in CARS `virtualenv`_ to check the code with these tools:

.. code-block:: console

    make lint

Use the following command to format the code with isort and black:

.. code-block:: console

    make format

Isort
-----
`Isort`_ is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

CARS ``isort`` configuration is done in `pyproject.toml`_

`Isort`_ manual usage examples:

.. code-block:: console

    cd CARS_HOME
    isort --check cars tests  # Check code with isort, does nothing
    isort --diff cars tests   # Show isort diff modifications
    isort cars tests          # Apply modifications

`Isort`_ messages can be avoided when really needed with **"# isort:skip"** on the incriminated line.

Black
-----
`Black`_ is a quick and deterministic code formatter to help focus on the content.

CARS ``black`` configuration is done in `pyproject.toml`_

If necessary, Black doesn’t reformat blocks that start with "# fmt: off" and end with # fmt: on, or lines that ends with "# fmt: skip". "# fmt: on/off" have to be on the same level of indentation.

`Black`_ manual usage examples:

.. code-block:: console

    cd CARS_HOME
    black --check cars tests  # Check code with black with no modifications
    black --diff cars tests   # Show black diff modifications
    black cars tests          # Apply modifications

Flake8
------
`Flake8`_ is a command-line utility for enforcing style consistency across Python projects. By default it includes lint checks provided by the `PyFlakes project <https://github.com/PyCQA/pyflakes>`_ , PEP-0008 inspired style checks provided by the `PyCodeStyle project <https://github.com/PyCQA/pycodestyle>`_ , and McCabe complexity checking provided by the `McCabe project <https://github.com/PyCQA/mccabe>`_. It will also run third-party extensions if they are found and installed.

CARS ``flake8`` configuration is done in `setup.cfg <https://raw.githubusercontent.com/CNES/cars/master/setup.cfg>`_

`Flake8`_ messages can be avoided (in particular cases !) adding "# noqa" in the file or line for all messages.
It is better to choose filter message with "# noqa: E731" (with E371 example being the error number).
Look at examples in source code.

Flake8 manual usage examples:

.. code-block:: console

  cd CARS_HOME
  flake8 cars tests           # Run all flake8 tests


Pylint
------
`Pylint`_ is a global linting tool which helps to have many information on source code.

CARS ``pylint`` configuration is done in dedicated `.pylintrc <//https://raw.githubusercontent.com/CNES/cars/master/.pylintrc>`_ file.

`Pylint`_ messages can be avoided (in particular cases !) adding "# pylint: disable=error-message-name" in the file or line.
Look at examples in source code.

Pylint manual usage examples:

.. code-block:: console

  cd CARS_HOME
  pylint tests cars       # Run all pylint tests
  pylint --list-msgs          # Get pylint detailed errors information



Jupyter notebooks
=================

CARS contains notebooks in tutorials directory.

To generate a `Jupyter kernel <https://jupyter.org/install>`_ with CARS installation, use:

.. code-block:: console

  make notebook

Follow indications to run a jupyter notebook.

Kernel is created with following command (with cars-version updated):

.. code-block:: console

  python -m ipykernel install --sys-prefix --name=cars-version --display-name=cars-version

To run the jupyter notebook, use:

.. code-block:: console

  jupyter notebook


Licensing
=========

When contributing to this project, ensure that any third-party tools or libraries integrated into your contribution are licensed under terms compatible with the Apache License, Version 2.0.
Specifically, the license of integrated tools must not impose restrictions that could "contaminate" or conflict with the Apache 2.0 license.
If you are unable to find a compatible license for a required tool, you may propose the contribution as an external plugin. External plugins allow the project to maintain its license integrity while still benefiting from your work. Always document the license of any external dependencies in your contribution.

Release and Version numbering
=============================

This project adheres to Semantic Versioning (`semver.org`_) to clearly communicate the impact of each release.
The version number follows the format MAJOR.MINOR.PATCH.
The first digit (MAJOR) is incremented when backward-incompatible changes are introduced, such as breaking changes to the high-level API.
The second digit (MINOR) is incremented when new features or modifications are added in a backward-compatible manner.
The third digit (PATCH) is reserved for backward-compatible bug fixes.
This approach ensures transparency and helps users understand the scope and impact of each update.



.. _`the GitHub repository`: https://github.com/CNES/cars
.. _`CARS Contribution guide`: https://github.com/CNES/cars/blob/master/CONTRIBUTING.md
.. _`virtualenv`: https://virtualenv.pypa.io/
.. _`Isort`: https://pycqa.github.io/isort/
.. _`Black`: https://black.readthedocs.io/
.. _`Flake8`: https://flake8.pycqa.org/
.. _`Pylint`: http://pylint.pycqa.org/
.. _`pyproject.toml`: https://raw.githubusercontent.com/CNES/cars/master/pyproject.toml
.. _`semver.org`: https://semver.org/

.. _user_manual_main_cli:

========
Main CLI
========

``cars`` is the unique entry point for CARS command line interface (CLI) to run 3D pipelines.

It enables two main pipelines : ``prepare`` and ``compute_dsm`` described in the following sections.

.. code-block:: console

    usage: cars [options] <command> [<args>]

    The cars commands are:
        prepare             Preparation for compute_dsm producing stereo-
                            rectification grid as well as an estimate of the
                            disparity to explore.
        compute_dsm         Tile-based, concurent resampling in epipolar geometry,
                            disparity estimation, triangulation and rasterization

    The options are :
      -h, --help            show this help message and exit
      --version, -v         show program's version number and exit

Arguments on a file
===================

Sometimes, for example when dealing with a particularly long argument lists, it may make sense to keep the list of arguments in a file rather than typing it out at the command line.
With CARS, the `@` character can be used to define a file containing arguments to be used.

.. code-block:: console

    cars @args.txt

Example of arguments list in @args.txt file:

.. code-block:: console

    prepare  -i data_samples/input12.json -o  data_samples/outprepare12 --nb_workers 2 --loglevel INFO

Examples files and data can be found in `demo directory <https://github.com/CNES/cars/tree/master/docs/source/demo>`_ in source code.

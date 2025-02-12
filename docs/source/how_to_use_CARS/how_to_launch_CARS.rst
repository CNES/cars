.. include:: ../links_substitution.rst

.. _how to launch CARS:

How to launch CARS
==================

``cars`` command line is the entry point for CARS to run the 3D pipeline.

.. code-block:: console

    cars -h

    usage: cars [-h] [--loglevel {DEBUG,INFO,PROGRESS,WARNING,ERROR,CRITICAL}] [--version] conf

    CARS: CNES Algorithms to Reconstruct Surface

    positional arguments:
      conf                  Inputs Configuration File

    optional arguments:
      -h, --help            show this help message and exit
      --loglevel {DEBUG,INFO,PROGRESS,WARNING,ERROR,CRITICAL}
                            Logger level (default: PROGRESS. Should be one of (DEBUG, INFO, PROGRESS, WARNING, ERROR, CRITICAL)
      --version, -v         show program's version number and exit

CARS cli takes only one ``.json`` file as command line argument:

.. code-block:: console

    cars configfile.json
    
Note that ``cars-starter`` script can be used to instantiate this configuration file.

.. code-block:: console

    cars-starter  -h

    usage: cars-starter [-h] -il [input.{tif,XML} ...] -out out_dir [--full] [--check]

    Helper to create configuration file
    options:
      -h, --help            show this help message and exit
      -il [input.{tif,XML} ...]
                              Input sensor list
      -out out_dir          Output directory
      --full                Fill all default values
      --check               Check inputs

Finally, an output ``used_conf.json`` file will be created on the output directory. This file contains all the parameters used during execution and can be used as an input configuration file to re-run cars.

Here is a reminder of what a basic json file configuration should look like :

.. sourcecode:: text

    {
        "inputs": {
            "sensors" : {
                "one": {
                    "image": "img1.tif",
                    "geomodel": "img1.geom",
                    "color": "color1.tif"
                },
                "two": {
                    "image": "img2.tif",
                    "geomodel": "img2.geom"
                }
            }
        },
        "orchestrator": {
          	"mode": "multiprocessing",
          	"nb_workers": 110,
          	"max_ram_per_worker": 6000
          	},
        "output": {
                "directory": "outresults"
        }
    }

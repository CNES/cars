
.. _configuration:

=============
Configuration
=============

This section describes main CARS configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file.

The structure follows this organisation:

.. sourcecode:: text

    {
        "pipeline": "sensor_to_full_resolution_dsm",
        "inputs": {
            ...
        },

        "applications": {
            "name_application_used_in_pipeline": {
                "parameter": value_parameter
            }
        }

        "orchestrator": {
            "mode": "local_dask"
        },

        "output": {
              "out_dir"="output_directory_path"
        }
    }
        
.. warning::

    Be careful with commas to separate each section. None needed for the last json element.


To have a full configuration, use :ref:`sensor_to_full_resolution_dsm_pipeline` section for main pipeline configuration examples.

Pipeline
========

This key is optional and defined the choice of CARS pipeline. At the moment, there are two of them:

* *sensor_to_full_resolution_dsm* as default, see :ref:`sensor_to_full_resolution_dsm_pipeline` for :ref:`sensor_to_full_resolution_dsm_pipeline_full_example`.
* *sensor_to_low_resolution_dsm*, for more advanced usage, a subpart of main pipeline to generate a low resolution DSM with sparse matches only.


.. _configuration_inputs:

Inputs
======

Values associated to this key are defined by pipeline so let's refer to :ref:`sensor_to_full_resolution_dsm_pipeline` for details.


Applications
============

This key is optional and allows to redefine parameters for each application defined in each "application_name" section used by pipeline.

See :ref:`applications` for details.

.. _orchestrator_config:

Orchestrator
============

This key is optional and allows to define orchestrator configuration that controls the distributed computations:

+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| Name             | Description                                               | Type                                    | Default value | Required |
+==================+===========================================================+=========================================+===============+==========+
| *mode*           | Parallelization mode "local_dask", "pbs_dask" or "mp"     | string                                  |local_dask     | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| *nb_workers*     | Number of workers                                         | int, should be > 0                      | 2             | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
| *walltime*       | Walltime for one worker                                   | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
+------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+


.. _configuration_outputs:

Outputs
^^^^^^^

Values associated to this key are defined by pipeline so let's refer to :ref:`sensor_to_full_resolution_dsm_pipeline`.


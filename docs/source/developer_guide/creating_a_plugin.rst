.. _creating_a_plugin:

=================
Creating a plugin
=================

.. _plugins:

Plugins
=======

This section describes optional plugin possibilities of CARS. 

.. note::
    
    Work in progress !

Plugins can be used to create new pipelines, overload applications in CARS pipelines or overload geometry functions

Installation
------------

To install a plugin, simply use pip inside your CARS environment :

.. code-block:: console
    
    source venv/bin/activate
    pip install plugin_name


Pipeline plugin
---------------

A pipeline plugin will add a new pipeline in CARS, with existing applications or new applications brought by the plugin.
If the plugin installed is a pipeline plugin, it can be used by specifying the "pipeline" parameter in your CARS configuration file.
For example :

.. code-block:: json

    {
        "pipeline": "name_given_by_plugin"
    }

New applications can be parametrized in "applications" section of the configuration file.

Application plugin
------------------

An application plugin will overload an existing application in CARS with a new method.
If the plugin installed is an application plugin, it can be used by adding the application in your CARS configuration file.
For example a point cloud denoising plugin can be used as follow :

.. code-block:: json

    {
        "applications": {
            "point_cloud_denoising": {
                "method": "name_given_by_plugin"
                "activated": true
            }
        }
    }

Geometry plugin
---------------

If the plugin installed is a pipeline plugin, it can be used by specifying the "geometry_plugin" parameter in your CARS configuration file.
For example :

.. code-block:: json

    {
        "geometry_plugin": "name_given_by_plugin"
    }

Sensor loader plugin
--------------------

Sensors loader plugins can be used by setting the "loader" parameter in "image" and "classification" attributes of the input configuration.
The loader defines several plugin-specific parameters

.. code-block:: json

    {
        "inputs": {
            "image": {
                "loader": "name_given_by_plugin"
                "plugin_specific_param1": "value1"
                "plugin_specific_param2": "value2"
            }
        }
    }

A sensor loader plugin is a class that overrides the `SensorLoaderTemplate` class. It must define two methods : 
 - check_conf : check the plugin-specific parameters
 - set_pivot_format : transform the input configuration into a configuration readable by CARS using pivot format and set it in class attribute `self.pivot_format`. Specifications of pivot format can be used on :ref:`advanced configuration`

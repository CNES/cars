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

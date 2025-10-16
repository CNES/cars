.. _config_overview:

Configuration overview
======================

This section describes CARS main configuration structure through a `yaml <http://www.yaml.org>`_ or `json <http://www.json.org/json-fr.html>`_ configuration file.

Only the configuration file is required to launch CARS with the command : 

.. code-block:: console

    cars config.yaml

A simple configuration file only needs two sections : ``input`` and ``output``. 
It is not recommended to use the other sections at first use : the default value of their parameters must work for your case.

The full structure follows this organization:

.. include-cars-config:: ../example_configs/configuration/organization_structure

Each section has a dedicated page where every parameter is documented.
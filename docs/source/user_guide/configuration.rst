
.. _configuration:

=============
Configuration
=============

This section describes main CARS configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file.

The structure follows this organisation:

.. sourcecode:: text

    {
        "inputs": ... ,
        "orchestrator": ... ,
        "applications": ... ,
        "output": ...
    }
        
.. warning::

    Be careful with commas to separate each section. None needed for the last json element.

.. toctree::
   :maxdepth: 2

   configuration/inputs.rst
   configuration/orchestrator.rst
   configuration/applications.rst
   configuration/outputs.rst
   configuration/full_example.rst




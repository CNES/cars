.. CARS documentation master file, created by
   sphinx-quickstart on Wed Sep  9 14:17:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: links_substitution.rst
	     
:Version: |version|

CARS, a satellite multi view stereo pipeline
============================================

:term:`CARS` is a dedicated and open source 3D software. It creates **Digital Surface Models** (:term:`DSM`) from pairs of satellite images using MultiView Stereo methods.
CARS is designed for massive production and scalability |cars_isprs|. It aims to be effectively used on HPC cluster as well as personal computers and is the core of the 3D image processing chain of the CO3D |co3d| satellite mission.

.. |img1| image:: images/animation_sat.gif
   :width: 100%
.. |img2| image:: images/overview_dsm_3d.gif
   :width: 80%

+--------------------+---------------------------------------------+
| From stereo images | CARS produces a Digital Surface Model (DSM) |
+--------------------+---------------------------------------------+
| |img1|             | |img2|                                      |
+--------------------+---------------------------------------------+

   
**Contact:** cars@cnes.fr

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   quick_start
   technical_foundations/index.rst

.. toctree::
   :caption: Configuration
   :maxdepth: 1

   configuration/config_overview.rst
   configuration/input.rst
   configuration/output.rst
   configuration/orchestrator.rst
   configuration/applications.rst
   configuration/advanced_parameters.rst
   configuration/full_config.rst

.. toctree::
   :caption: Examples
   :maxdepth: 1

   general_examples/index.rst
   sensor_examples/index.rst

.. toctree::
   :caption: Developer guide
   :maxdepth: 1

   developer_guide/concepts/software_conception
   developer_guide/algorithm_conception
   developer_guide/contributing_to_cars
   developer_guide/creating_a_plugin


.. toctree::
   :caption: References
   :maxdepth: 1

   glossary
   bibliography
   api_reference/index.rst
   related_software
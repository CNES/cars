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

.. warning::
  
   CARS maturity is tightly linked to CO3D image processing chain development. 
   As so, version ``1.0.0`` of CARS would be the last upgraded version of CARS for CO3D and every version released before that shall be considered experimental.
   
**Contact:** cars AT cnes.fr

.. toctree::
   :caption: Content
   :maxdepth: 2

   getting_started
   exploring_the_field/index.rst
   usage/index.rst
   recipes
   software_design/index.rst
   troubleshooting_and_faqs
   contributing_the_project


.. toctree::
   :caption: Tutorials
   :maxdepth: 1
	     
   A high scalability satellite 3D stereo reconstruction framework with AI enhancement <https://github.com/cars-cnes/3d-stereo-with-ai>
   Discover CNES 3D open-source tools through a realistic scenario <https://github.com/cars-cnes/discover-cnes-3d-tools#discover-cnes-3d-open-source-tools-through-a-realistic-scenario>

   
.. toctree::
   :caption: References
   :maxdepth: 1

   glossary
   bibliography
   api_reference/index.rst


.. toctree::
   :caption: Third parties
   :maxdepth: 1
   	      
   PANDORA, stereo matching <https://github.com/cnes/pandora>
   SHARELOC, geometric library <https://github.com/cnes/shareloc>
   DEMCOMPARE, dem analysis <https://github.com/cnes/demcompare>
   BULLDOZER, dsm to dtm <https://github.com/cnes/bulldozer>
   CARS-RESAMPLE, grid resampling <https://github.com/cnes/cars-resample>
   CARS-RASTERIZE, point cloud to dsm <https://github.com/cnes/cars-rasterize>
   CARS-MESH, surface reconstruction <https://github.com/cnes/cars-mesh>

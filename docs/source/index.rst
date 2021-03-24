.. CARS documentation master file, created by
   sphinx-quickstart on Wed Sep  9 14:17:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CARS's documentation!
================================

CARS is a dedicated and open source 3D tool to produce Digital Surface Models from satellite imaging by photogrammetry.

This Multiview stereo pipeline is intended for massive DSM production with a robust and performant design.

CARS means CNES Algorithms to Reconstruct Surface (or Chaîne Automatique de Restitution Stéréoscopique in french)

It is composed of:
A Python API, based on xarray, enabling to realize all the computation steps leading to a DSM.
An end-to-end processing chain based on this API. It can be performed using dask (locally or on a cluster which has a GPFS centralized files storage) or multiprocessing libraries to distribute the computations.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   generalities
   install
   cli_usage
   notebooks
   modules

Indices and tables
==================

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`

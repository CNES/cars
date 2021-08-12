=================
CARS loaders
=================


CARS modularities aims at:

- Be able to load several geometry libraries
- Be able to load several matching libraries
- Be able to use several opening image tools
- Be able to use several cluster scheduling strategy/tools

The goals are:

- answer several project and user goals.
- ease 3D studies and definition/test of new algorithms
- simplify evolution when a library is becoming obsolete

In CARS, we call "loader" the generic way to include

The loaders implementations can be internal or external.

General loader design
=====================

Define how the loader are used in CARS.

- Loader abstract class
- Loader externalization possibility (registration)
- How to call/use the abstract class
- Link with configuration dynamic registration.

The several loader types below are declined from this general way.

Geometry loader
===============

This geometry loader aims to be able to use several geometry libraries, typically OTB, libGEO and shareloc.

Describe here how to interact with geometric models in CARS design.



Matching loader
===============

TODO

Cluster loader
==============

TODO

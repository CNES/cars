=================
CARS Requirements
=================

Modularity
----------

Here are several needs for future CARS:
- Geometric core library : be able to have OTB, shareloc, libgeo, ... internal and by plugins
- Matching step : be able to call several matching tool. Pandora has to be called in a generic way with clean API.
- Input data library : be able to input several type of images. Only rasterio possible ? or plugins also here ?

etc ...

This modularity has to be well designed and document so as another developer can easily add other possibilities.

Another modularity is the possibility to include other code between steps.
Maybe with  the possibility to change static call graph ?  and sub functions between steps ?
Maybe with some possibilities to add plugins in pipeline between steps ?

Naming of the modularities.

Shareable
---------
The software has to be easily shareable and developed by other people.
Needs clean design, documentation, examples, notebooks, ...


Others ?

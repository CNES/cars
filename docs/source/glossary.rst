.. _glossary:

========
Glossary
========

3D CARS common words and shortened terms are detailed here.

To update, follow `glossary sphinx documentation`_ in RST source documentation.

.. glossary::
    CARS
      means CNES Algorithms to Reconstruct Surface (ou Chaîne Automatique de Restitution Stéréoscopique en français)
     
    CNES
      Centre National d'Etudes Spatiales
      
    core
      Means the internal CARS function used as the engine core for steps and pipelines.

    disp
      The short version of "disparity"

    disparity
      The column difference between a pixel in the left image and its homologous pixel in the right image.

    DEM
      `Digital Elevation Model`_. Usually means all elevation models in raster: DSM, DTM,...

    DSM
      Digital Surface Model. Represents the earth's surface and includes all objects on it.
      CARS generates DSMs. See `Digital Elevation Model`_

    DTM
      Digital Terrain Model. Represents bare ground surface without any objects like plants and buildings
      You need another tool to generate DTM from CARS DSM. See `Digital Elevation Model`_

    epi
      A shortened version for "epipolar". Simplify length of functions in CARS code.

    epipolar
      Refer to `epipolar geometry`_ used as basis for CARS 3D pipeline.

    matching
      Stereo matching or disparity estimation is the process of finding the pixels
      in the multiscopic views that correspond to the same 3D point in the scene.

    OTB
      `Orfeo Toolbox <https://www.orfeo-toolbox.org/>`_ is an open-source project for state-of-the-art remote sensing applications.

    pipeline
      In computing, a pipeline, also known as a data pipeline is a set of data
      processing elements connected in series, where the output of one element
      is the input of the next one. In CARS, pipeline orchestrates applications/functions
      to chain 3D steps to produce DSM.

    rectification
      `Image rectification`_ is a transformation process used to project images onto a common image plane.
      In CARS, the epipolar geometry rectification is used.

    ROI
      `Region of Interest`_ means a subpart of the `DSM` raster in CARS.
      It can be defined by a file or a bounding box.
     
    RPC
      Rational Polynomial Coefficient
      
    SIFT
      Scale-Invariant Feature Transform



.. _`Digital Elevation Model`: https://en.wikipedia.org/wiki/Digital_elevation_model
.. _`Digital Surface Model`: https://en.wikipedia.org/wiki/Digital_elevation_model
.. _`epipolar geometry`: https://en.wikipedia.org/wiki/Epipolar_geometry
.. _`Image rectification`: https://en.wikipedia.org/wiki/Image_rectification
.. _`Region of Interest`: https://en.wikipedia.org/wiki/Region_of_interest

.. _`glossary sphinx documentation`: https://sublime-and-sphinx-guide.readthedocs.io/en/latest/glossary.html

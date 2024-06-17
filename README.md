<div align="center">
<a target="_blank" href="https://github.com/CNES/cars">
<picture>
  <source
    srcset="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_dark.png"
    media="(prefers-color-scheme: dark)"
  />
  <img
    src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/picto_light.png"
    alt="CARS"
    width="40%"
  />
</picture>
</a>

<h4>CARS, a satellite multi view stereo framework </h4>

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Documentation](https://readthedocs.org/projects/cars/badge/?version=latest)](https://cars.readthedocs.io/?badge=latest)

<p>
  <a href="#overview">Overview</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#references">References</a>
</p>
</div>

## Overview

From stereo images  |  CARS produces a Digital Surface Model (DSM)
:-------------------------:|:-------------------------:
<img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/animation_sat.gif" alt="drawing" width="100%"/> |  <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/overview_dsm_3d.gif" alt="drawing" width="100%"/>


**CARS** is an open source 3D tool dedicated to produce **Digital Surface Models** from satellite imaging by photogrammetry.
This Multiview Stereo framework is intended for massive DSM production with a robust, performant and modular design.

Be aware that the project is new and is evolving to maturity with CNES usage roadmaps and projects such as:
- <a href="https://co3d.cnes.fr/en/co3d-0">CO3D project &nbsp;&nbsp;&nbsp;  <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo_co3D_cnes.jpg" height="20"/></a>
- <a href="https://www.ai4geo.eu">AI4GEO project &nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo-ai4geo.png" height="20"/> </a>
- <a href="https://www.evo-land.eu">Evoland project &nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo-evoland.png" height="20"/> </a>

## Quick start

### CARS Docker Image

[![Docker Status](http://dockeri.co/image/cnes/cars)](https://hub.docker.com/r/cnes/cars)

CARS is available on Docker Hub and can be downloaded by:
``` bash
docker pull cnes/cars
```

### One main pipeline to generate DSM

You only need to launch one command:

```
docker run -w /data -v "$(pwd)"/data_gizeh:/data cnes/cars /data/configfile.json
```

with one configuration input file ("configfile.json") located in a "data" folder to be consistent with the previous command lines:
```
{

        "inputs": {
            "sensors" : {
                "one": {
                    "image": "img1.tif",
                    "geomodel": "img1.geom"
                },
                "two": {
                    "image": "img2.tif",
                    "geomodel": "img2.geom"
                },
                "three": {
                    "image": "img3.tif",
                    "geomodel": "img3.geom"
                }
            },
            "pairing": [["one", "two"],["one", "three"]],
            "initial_elevation": "srtm_dir/N29E031_KHEOPS.tif"
        },

        "output": {
              "out_dir": "outresults"
        }

}

```

### On the way to the Pyramids...

You want to build the pyramids by yourself? Download our [open licence](https://www.etalab.gouv.fr/licence-ouverte-open-licence) Pleiades [data sample](https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2) to give CARS a try!

## Documentation

Go to [CARS Main Documentation](https://cars.readthedocs.io/?badge=latest).


## Contribution

To do a bug report or a contribution, see the [**Contribution Guide**](https://github.com/CNES/cars/blob/master/CONTRIBUTING.md).

For project evolution, see [**Changelog**](https://github.com/CNES/cars/blob/master/CHANGELOG.md).

## Credits

See [Authors file](https://github.com/CNES/cars/blob/master/AUTHORS.md)


## References

- [Youssefi D., Michel, J., Sarrazin, E., Buffe, F., Cournet, M., Delvit, J., L’Helguen, C., Melet, O., Emilien, A., Bosman, J., 2020. Cars: A photogrammetry pipeline using dask graphs to construct a global 3d model. IGARSS - IEEE International Geoscience and Remote Sensing Symposium.](https://ieeexplore.ieee.org/document/9324020)
- [Michel, J., Sarrazin, E., Youssefi, D., Cournet, M., Buffe, F., Delvit, J., Emilien, A., Bosman, J., Melet, O., L’Helguen, C., 2020. A new satellite imagery stereo pipeline designed for scalability, robustness and performance. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/171/2020/)

<div align="center">
<a target="_blank" href="https://github.com/CNES/cars">
<picture>
  <img
    src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/cars_picto.svg"
    alt="CARS"
    width="50%"
  />
</picture>
</a>

<h4>CARS, a satellite multi view stereo framework </h4>

[![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Documentation](https://readthedocs.org/projects/cars/badge/?version=latest)](https://cars.readthedocs.io/?badge=latest)
[![Github Action](https://github.com/CNES/cars/actions/workflows/cars-ci.yml/badge.svg?branch=master)](https://github.com/CNES/cars/actions)
[![pypi](https://badge.fury.io/py/cars.svg)](https://pypi.org/project/cars/)
[![Docker pulls](https://img.shields.io/badge/dynamic/json?formatter=metric&color=blue&label=Docker%20pull&query=%24.pull_count&url=https://hub.docker.com/v2/repositories/cnes/cars)](https://hub.docker.com/r/cnes/cars)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17406459.svg)](https://doi.org/10.5281/zenodo.17406459)

<p>
  <a href="#overview">Overview</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#help">Help</a> •
  <a href="#credits">Credits</a> •
  <a href="#contact">Contact</a> •
  <a href="#references">References</a>
</p>
</div>

# 🌐 **Overview**

<div align="center">
<table style="display: inline-block;">
  <tr>
    <td align="center">
      <div  style="font-size:2em; margin-bottom:10px;"><strong>🎞️ From Stereo Images</strong></div><br>
      <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/animation_sat.gif"
           alt="Stereo images animation" width="95%"><br>
      <em>Input stereo image pairs</em>
    </td>
    <td align="center">
      <div  style="font-size:2em; margin-bottom:10px;"><strong>🌍 To Digital Surface Model (DSM)</strong></div><br>
      <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/overview_dsm_3d.gif"
           alt="DSM overview" width="95%"><br>
      <em>Output 3D surface reconstruction</em>
    </td>
  </tr>
</table>
</div>

**CARS** is an open source 3D tool dedicated to produce **Digital Surface Models** from satellite imaging by photogrammetry.
This Multiview Stereo framework is intended for massive DSM production with a robust, performant and modular design.

**CARS** is currently under active development and integrated into various projects & missions:

<div align="center">
<p align="center">
  <a href="https://co3d.cnes.fr/en/co3d-0" target="_blank">
    <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo_co3D_cnes.png"
         alt="CO3D project" height="70">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.ai4geo.eu" target="_blank">
    <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo-ai4geo.png"
         alt="AI4GEO project" height="75">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.evo-land.eu" target="_blank">
    <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/logo-evoland.png"
         alt="Evoland project" height="60">
  </a>
</p>
</div>

<br>

Its roadmap and implemented functionalities are evolving regularly depending on specific project requirements.

# 🚀 **Quick start**


### △ On the way to the Pyramids...
---

You want to build the pyramids by yourself? 


<div align="center">
<table style="display: inline-block;">
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/dsm.png"
           alt="Dsm" width="400"><br>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/clr.png"
           alt="Color" width="400"><br>
    </td>
     <td align="center">
      <img src="https://raw.githubusercontent.com/CNES/cars/master/docs/source/images/dsm_clr.png"
           alt="Color" width="400"><br>
    </td>
  </tr>
</table>
</div>

Download our [open licence](https://www.etalab.gouv.fr/licence-ouverte-open-licence) Pleiades [data sample](https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2) to give CARS a try!


### 🐋 with Docker
---

#### 🛠 Installation

CARS is available on Docker Hub and can be downloaded by:
``` bash
docker pull cnes/cars
```

#### ⚙ Run CARS

You only need to launch one command:

```
docker run -w /data -v "$(pwd)"/data_gizeh:/data cnes/cars /data/configfile.yaml
```

### 🐍 with pip
---

#### 🛠 Installation

CARS can also be downloaded using the pip install command:

``` bash 
pip install cars 
```

#### ⚙ Run CARS

Once you moved to the data_gizeh directory:

``` bash 
cars configfile.yaml
```

# 📚 **Documentation**

Go to [CARS Main Documentation](https://cars.readthedocs.io/?badge=latest).

# 📜 **License**

CARS is licensed under [Apache License v2.0](https://www.apache.org/licenses/LICENSE-2.0). Please refer to the [LICENSE](https://gitlab.cnes.fr/dali/cars-park/cars/-/blob/1214-mise-a-jour-du-readme-pour-la-nouvelle-version-1-0-0/LICENSE) file for more details.

# 🤝 **Contribution**

To do a contribution, see the [Contribution Guide](https://github.com/CNES/cars/blob/master/CONTRIBUTING.md).  For project evolution, see [Changelog](https://github.com/CNES/cars/blob/master/CHANGELOG.md).

# 🆘 **Help**

For issues, questions, or feature requests, please open an issue on our [GitHub Issues](https://github.com/CNES/cars/issues) page or check the [documentation](https://cars.readthedocs.io/en/stable/index.html) for additional resources.

You can also ask your questions on the corresponding [slack](https://join.slack.com/t/cars-community/shared_invite/zt-2uw6usog1-~TT~m8BxO9faMXpP7tpz2Q).

# ✒️ **Credits**

If you use CARS in your research, please cite the following paper:

```
@INPROCEEDINGS{9324020,
  author={Youssefi, David and Michel, Julien and Sarrazin, Emmanuelle and Buffe, Fabrice and Cournet, Myriam and Delvit, Jean-Marc and L'Helguen, Céline and Melet, Olivier and Emilien, Aurélie and Bosman, Julien},
  booktitle={IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={CARS: A Photogrammetry Pipeline Using Dask Graphs to Construct A Global 3D Model}, 
  year={2020},
  volume={},
  number={},
  pages={453-456},
  keywords={Automobiles;Geometry;Three-dimensional displays;Computational modeling;Estimation;Stereo image processing;Task analysis;Image processing;Photogrammetry;Stereovision;Digital Surface Model;High Performance Computing;Cloud Computing;Dask},
  doi={10.1109/IGARSS39084.2020.9324020}}
```

See [Authors file](https://github.com/CNES/cars/blob/master/AUTHORS.md)

# ✉️ **Contact**

You can contact us on the following mail adress: cars@cnes.fr

# 🔗 **References**

- [Youssefi D., Michel, J., Sarrazin, E., Buffe, F., Cournet, M., Delvit, J., L’Helguen, C., Melet, O., Emilien, A., Bosman, J., 2020. Cars: A photogrammetry pipeline using dask graphs to construct a global 3d model. IGARSS - IEEE International Geoscience and Remote Sensing Symposium.](https://ieeexplore.ieee.org/document/9324020)
- [Michel, J., Sarrazin, E., Youssefi, D., Cournet, M., Buffe, F., Delvit, J., Emilien, A., Bosman, J., Melet, O., L’Helguen, C., 2020. A new satellite imagery stereo pipeline designed for scalability, robustness and performance. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/171/2020/)

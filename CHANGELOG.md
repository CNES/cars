# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is set to the next chosen version, the name of the milestone and the date (month year).
A new section Unreleased is opened then for next dev phase.

## Unreleased

### Added

### Changed

### Fixed

## 0.5.1 Minor fixes post refactoring (September 2022)

### Changed

- New User Guide version [#448]

### Fixed

- Fix dask upgrade bug [#436]
- Fix temporarily local path to CARS OTB installed apps [#434]
- Fix multiprocessing bug [#446]
- Fix (remove) warning: tile ignored [#455]

## 0.5.0 Refacto CARS (August 2022)

Be careful, this version changes API and configuration ! See new documentation.

### Added

- Add documentation generation pre-commit when git push [#406]
- Add docker images files for jupyter and cars tutorial [#418]
- Add cars tutorial main_tutorial as notebook slides [#421]
- Add pandora mccn simplified installation process [#426]

### Changed

- Refactoring of all CARS code by implementing new concepts as application, CarsDataset, orchestrator, pipeline and configuration. See documentation for more details. [#332]
- upgrade support to python 3.8 only [#372]
- change quick_start and notebooks to tutorials structure [#419]

### Fixed

- remove setuptools pre installation [#341]
- remove click 8.0.4 dependency [#341]
- add automatic use of setup.cfg docs extra_require for readthedocs conf [#367]


## 0.4.2 OTB/GDAL upgrade (July 2022)

### Fixed

- Fix gdal-config dependency in docker with rasterio upgrade [#404]
- Upgrade OTB to 7.4.1 in Dockerfile [#404]


## 0.4.1 Stabilization - Upgrade Pandora 1.2.* (June 2022)

### Added

- Add Authors file for contributors [#400]

### Changed

- Upgrade to pandora 1.2.* [#371]
- Improve performance of CARS cli with no argument [#357]
- Add egm96 internal geoid hdr header [#335]
- Cluster dask code structure evolution [#355]
- Clean : remove hard coded indexes in the cloud fusion [#394]
- Gitlab template evolution [#399]
- Makefile evolution and clean [#391]

### Fixed

- Fix min_elevation_offset option when user_disp_min is greater than disp_min [#348]
- Temporary bugfix by forcing click version to avoid dask.distributed pbs cluster trouble [#383]
- Fix CI errors with pylint upgrades and docker apt package ubuntu upgrades  [#390, #388, #385, #384, #380, #379, #376, #369, #368, #339]
- Monitor pytest CI performance with debug information [#373]
- Fix shapely version upgrade bug [#349]
- Fix pygdal setuptools version upgrade bug [#333]
- Fix OTB geometry plugin bugs [#393, #396]
- Fix Loader geometry bug: non static schemas management [#395]
- Filter useless verbose dask warning [#353]


## 0.4.0 Geometry Loader (September 2021)

### Added

- Write used dask configuration in output directory. [#224]
- Add argparse file option @"opts.txt" + doc [#265]
- Add Contributor Licence Agreement [#257]
- Add quality code tools isort, black, flake8 [#247, #269, #271]
- Add prepare/compute_dsm notebook [#246]
- Add sonarqube configuration [#198]
- Add Geometry Loader mechanism [#287]
- Add OTBGeometry loader [#287]

### Changed

- Update/Clean package setup (add Makefile, clean requirements) [#210, #193, #305, #197]
- Make pip install -e work [#207]
- Update/Clean environment parameters [#166, #251]
- Move cars_cli.py from bin to cars. [#188, #203]
- Rename cars_cli to cars. [#188]
- Change default nb_workers to 2 [#218]
- Allow multiprocessing fork mode. [#283]
- Force OpenMP use in dask, and TBB in multiprocessing. [#304]
- Change loglevel argument API to pipeline level [#310, #311]
- Upgrade and fix pandora dependency [#235, #267, #274, #273, #309, #188]
- Clean quality code pylint and sonarqube conf [#302, #209]
- Change CARS loglevel default to WARNING + clean output [#239, #300, #143]
- Refactoring CARS file/module organization [#216, #259]
- Clean Documentation + ReadTheDocs [#160, #258]
- Dockerfile update [#219]
- Update/Clean README [#200]
- Geoid file indicated via the static configuration [#287]

### Fixed

- Fix epipolar size. [#206, #237, #248]
- Fix and clean tbb support [#267, #276, #304]
- Fix align_with_lowres_dem in mp mode [#286]
- Fix Delaunay algorithm between epi and terrain tiles [#277]
- Fix dask ComputeDSMMemoryLogger api [#202]
- Handle margins when setting the disparity to 0 [#201]
- Fix car cli with setuptools_scm version [#199]

## 0.3.0 Multi-classes mask management (December 2020)

### Added

- Mask management : change input format and internal behaviors. [#147, #170]
- Constants.py added to optimize code [#172]
- Default elevation option [#111]
- Satellite angles information on prepare step [#58, #190]
- Changelog added [#185]

### Changed
- Update rasterization tests [#177]
- Integrate a new memory estimation method [#158]
- Version handling with setuptools_scm [#194]

### Fixed
- Fix CLI parsing for ROI option : **CLI API changes** (-i for input json file and -o for output directory) [#93, #174]
- Pylint code clean [#191]
- Add constants.py file [#172]

## 0.2.0 First Open Source Official Release (July 2020)

- 3D functional pipeline : Python3 and C++ OTB modules
- Python API
- CLI command
- Documentation basics
- Continuous Integration on unit tests

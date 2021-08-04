# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is set to the next chosen version, the name of the milestone and the date (month year).
A new section Unreleased is opened then for next dev phase.

## Unreleased

### Added

- Write used dask configuration in output directory. [#224]
- Add argparse file option @"opts.txt" + doc [#265]
- Add Contributor Licence Agreement [#257]
- Add quality code tools isort, black, flake8 [#247, #269, #271]
- Add prepare/compute_dsm notebook [#246]
- Add sonarqube configuration [#198]

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

- 3D functionnal pipeline : Python3 and C++ OTB modules
- Python API
- CLI command
- Documentation basics
- Continuous Integration on unit tests

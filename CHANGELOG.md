# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features. 
When publication of a new release, the section "Unreleased" is blocked to the next chosen version and name of the milestone at a given date. 
A new section Unreleased is opened then for next dev phase. 


## Unreleased 

### Added


### Changed 

- Move cars_cli.py from bin to cars. [#188]
- Rename cars_cli to cars. [#188]


### Fixed


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



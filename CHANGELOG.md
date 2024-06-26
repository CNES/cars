# Changelog

## 0.8.0  Orfeo Toolbox dependency removal (June 2024)

### Added

 - Auto mode for orchestrator adapts to machine resources [#798]

### Changed

 - Orfeo Toolbox geometry plugin is no longer available [#805]
 - Exogenous DEM is cropped before disparity grid generation to reduce time and memory consumption [#827]

### Fixed

 - Subpixellic shift during resampling is fixed [#799]
 - Input mask can be saved as output of dense_point_clouds_to_dense_dsm pipeline [#816]
 - Ambiguity is normalized globally and no longer by tiles [#810]
 - Classified pixels outside of epipolar footprint are no longer filled [#814]
 - Statistics on confidence intervals are added and warnings are removed [#817]
 - Parameter check_inputs can be used with Shareloc geometry plugin [#820]
 - Masked and filled pixels are no longer masked in final DSM [#836]
 - Cache of GDAL is limited to 500Mb to avoid memory overload during resampling [#823]

## 0.7.6  Confidence intervals (April 2024)

### Added

 - Integration of Pandora confidence intervals [#764]
 - Option to choose endogenous or exogenous DEM to resample images [#790]
 - Geomodel parameter is optional [#785]
 - Path to JSON file is allowed for dense matching loader configuration [#552]
 - Support for point cloud denoising plugin [#771]
 - Check footprint of given initial elevation [#780]

 ### Changed

  - Version 0.2.0 of Shareloc is used : fast grid generation [#781]

 ### Fixed

  - Invert rows and columns for generation of disparity range grids with Shareloc [#804]
  - Better typing of output files [#758]
  - Faster LAZ point cloud saving [#795]
  - Avoid computation of useless epipolar tiles when a ROI is given for no_merging pipelines [#791]
  - Compute footprint of endogenous DEM according to images and epipolar footprint [#780]
  - Broadcast delayed objects with dask to avoid memory overload [#796]

## 0.7.5  Pandora integration (March 2024)

### Added

 - New default pipeline sensors_to_dense_dsm_no_merging with cumulative rasterisation [#698]
 - Profiling report for multiprocessing mode [#745]
 - Dashboard showing tiles processing during a run [#765]
 - Save of sparse matches in sensor geometry [#761]
 - Relaunch of frozen tasks after timeout [#768]

### Changed

 - Version 1.6.0 of Pandora is used [#688]
 - Version 0.2.0a2 of Shareloc is used [#756]
 - Sparse matching processed by strip to improve computation time [#753]
 - DEM generation improved : less artifacts, filling of invalid regions [#754]
 - Default orchestrator is multiprocessing [#755]
 - Disparity interval cropped when it is too high [#757]

### Fixed

 - Data is typed to save memory consumption [#661]
 - Preload of CARS in workers for multiprocessing mode to reduce forking time [#730]
 - CPU usage limited to 100% per worker [#750]
 - Output classification pixels cannot be floating-point numbers anymore
 - ROI not used for endogeneous DEM [#772]



## 0.7.4  Local disparity ranges (December 2023)

### Added

- Local disparity ranges are used [#695]
- Write point cloud by pair [#727]
- Left and right classification fusion on disparity map [#729]

### Changed

- Plane filling method: no propagation by default [#725]

### Fixed

- Slurm cluster launch [#736]
- None tiles managed in save_cars_dataset [#716]
- Color data type propagated to the end [#642]
- Memory unreleased in multiprocessing mode [#731]
- Loglevel could not be set [#734]



## 0.7.3  CARS fully pip installable (November 2023)

### Added

- CARS can now be pip installed [#639]
- Name of color bands propagated in point cloud [#696]

### Changed

- Slurm account now as a parameter of Orchestrator [#712]
- pkg_resources not used anymore [#700]

### Fixed

- Rasterization in epsg:4326 supported [#724]
- Estimation of DSM roi in Shareloc plugin [#701]

## 0.7.2 Self calculated initial elevation (October 2023)

### Added
- Multiprocessing cluster: add factorization of Delayed [#673]
- 8 bits images are managed with particular sift parameters [#668]
- Self calculated initial elevation with sift matches [#664]
- Add debug mode with roi using initial tiling [#475]
- Information map about dense matches filling [#679]
- Script to generate CARS configuration [#678]

### Changed
- Multi points direct location [#708]
- Application parameter "method" now optional [#692]
- Geometry plugin interface [#623]
- Nodata parameter now optional [#296]

### Fixed
- Orchestrator used does not match the one reported in the logs [#681]
- used_conf is no longer created when the application is initialized [#693]
- Parameter "mask" can be used in "dense_point_clouds_to_dense_dsm" [#722]

## 0.7.1 Slurm Cluster (July 2023)

### Added
- Dask Slurm cluster [#624]
- Crop disparity range with thresholds [#626]
- Used source information map generated [#648]

### Changed
- Write workers logs in one single file [#659]
- Use tempfile to generate default orchestrator folder instead of /tmp [#663]

### Fixed
- Cython upgrade bug [#674]
- Too much used disk memory in MP mode [#632]


## 0.7.0 CARS installable without OTB (June 2023)

### Added
- Notebook: add masks and classification options [#615]
- Option: Set python interpreter to use in PBS Dask Cluster [#611]
- DensePointCloudToDenseDSM Pipeline: now re-entrance with confidence and classification [#602]
- Shareloc Geometry plugin is now internal [#618]
- Use_sec_disp option allowing to use right disparity map is removed [#638]
- CARS can now be installed without OTB [#637]
- Epipolar tiles corresponding to region outside sensor images are removed before any computation [#635]
- Add CARS progress messages [#658]

### Changed
- Multiprocessing cluster doesn't freeze when bug occur anymore [#607]
- Resampling is not done with OTB anymore [#508]
- Pyproj is no longer constrained [#646]

### Fixed
- Notebook: retrieve dsm [#614]
- Doc: Roi parameter disappearing [#616]


## 0.6.0 A new input interface (mask and classification) and new pipelines (March 2023)

### Added
- All confidence maps from Pandora are propagated [#521]
- New pipelines: sensor to point cloud, point cloud to dsm[#567]
- Generate performance map [#590]
- Density parameter for terrain tile size computation in pc to dsm pipeline [#566]
- Multiprocessing mode available for DenseMatchesFilling [#539]
- Rasterization coded in C++ [#523]

### Changed
- P+XS Fusion is no longer available in CARS [#579]
- Input mask is now a binary validity mask used in the whole pipeline [#566]
- Region of Interest with GeoJson or Shapefile [#580, #242]
- Refactoring of Masks, Classifications and Color [#577, #578]
- Move the former "set_to_ref_alt" option to DenseMatchesFilling application [#577, #578]

### Fixed
- Bug DenseMatchesFilling Plane [#599]


## 0.5.4 Grid correction in the configuration file and a better estimate of disparity range (February 2023)

### Added
- A better estimate of disparity range [#538]
- Prepare and use Epipolar A Priori data: grid correction and disparity range [#458]
- Use custom DASK configuration for CARS [#554]
- Use adaptative DASK cluster [#541]
- Update Pandora to 1.3 [#459][#575][#587]
- Point cloud saving in the low resolution pipeline [#498]

### Changed
- Update black formatting rules [#556]
- Update dask memory logger, disable it by default [#527]
- OTB_MAX_RAM_HINT not used for tile sizing anymore [#541]
- Deactivate DASK dashboard by default [#343]

### Fixed
- Line artefact on right disparity map [#525]
- Pipeline configuration override [#533] 
- Scipy an xarray broken interfaces [#561]
- Set masked pixels to nodata in point clouds from triangulation [#465]
- Set log level of dask cluster to warning [#545]


## 0.5.3 Points clouds as LAZ files (December 2022)

### Added
- Dump point clouds as LAS [#438]
- Refactoring of tiling grids and margins [#451]
- Generate a runnable full configuration file [#422]
- Add hole filling application for disparity maps [#275]

### Changed
- Add minimum number of sift matches in configuration [#514]

### Fixed
- Fix pylint errors (no-member) in application template [#437]
- Fix bug PBS Dask [#515]
- Fix wrong images saved in notebooks
- Fix bug in disparity map saving [#484]


## 0.5.2 Profiling and dense matching confidence information (November 2022)

### Added
- Add ambiguity transmission through pipeline [#478]
- Add memory/cpu profiling mode [#473]
- Clean notebooks outputs [#497]

### Changed
- Clean setup.cfg with optionnal package data [#467]
- Clean disk data in Multiprocessing mode [#454]

### Fixed
- Fix dimension bug during point cloud fusion [#482]


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
- Remove setuptools pre installation [#341]
- Remove click 8.0.4 dependency [#341]
- Add automatic use of setup.cfg docs extra_require for readthedocs conf [#367]


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

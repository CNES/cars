# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
# Dependencies : python3 venv internal module
# Recall: .PHONY  defines special targets not associated with files
#
# Some Makefile global variables can be set in make command line:
# CARS_VENV: Change directory of installed venv (default local "venv" dir)
# LOGLEVEL: pytest LOGLEVEL (default INFO)

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: CARS_VENV="other-venv/" make install
ifndef CARS_VENV
	CARS_VENV = "venv"
endif
CARS_VENV := $(abspath $(CARS_VENV))

# Set pytest LOGLEVEL if not defined in command line
# Example: LOGLEVEL="DEBUG" make test
ifndef LOGLEVEL
	LOGLEVEL = "INFO"
endif

# Check CMAKE, OTB variables before venv creation
CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_OTB = $(shell command -v otbcli_ReadImageInfo 2> /dev/null)

# Check python install in VENV
CHECK_NUMPY = $(shell ${CARS_VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${CARS_VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${CARS_VENV}/bin/python -m pip list|grep rasterio)
CHECK_TBB = $(shell ${CARS_VENV}/bin/python -m pip list|grep tbb)
CHECK_NUMBA = $(shell ${CARS_VENV}/bin/python -m pip list|grep numba)
CHECK_CYVLFEAT = $(shell ${CARS_VENV}/bin/python -m pip list|grep cyvlfeat)
TBB_VERSION_SETUP = $(shell cat setup.cfg | grep tbb |cut -d = -f 3 | cut -d ' ' -f 1)

# Check Docker
CHECK_DOCKER = $(shell docker -v)

# CARS version from setup.py
CARS_VERSION = $(shell python3 -c 'from cars import __version__; print(__version__)')
CARS_VERSION_MIN =$(shell echo ${CARS_VERSION} | cut -d . -f 1,2,3)

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      CARS MAKE HELP  LOGLEVEL=${LOGLEVEL}"
	@echo "  Dependencies: Install OTB and VLFEAT before !"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: check
check: ## check if VLFEAT is installed
	@[ "${VLFEAT_INCLUDE_DIR}" ] || ( echo ">> VLFEAT_INCLUDE_DIR is not set (see make vlfeat)"; exit 1 )
	@[ "${VLFEAT_LIBRARY_DIR}" ] || ( echo ">> VLFEAT_LIBRARY_DIR is not set (see make vlfeat)"; exit 1 )

.PHONY: venv
venv: check ## create virtualenv in CARS_VENV directory if not exists
	@test -d ${CARS_VENV} || python3 -m venv ${CARS_VENV}
	@${CARS_VENV}/bin/python -m pip install --upgrade "pip<=23.0.1" setuptools # no check to upgrade each time
	@touch ${CARS_VENV}/bin/activate

.PHONY: vlfeat
vlfeat: ## install vlfeat cnes fork library locally
	@test -d vlfeat || git clone https://github.com/CNES/vlfeat.git
	@cd vlfeat && make MEX=$MATLABROOT/bin/
	@echo "vlfeat is installed. Please set the following environment variables:"
	@echo "export VLFEAT_INCLUDE_DIR=${PWD}/vlfeat"
	@echo "export VLFEAT_LIBRARY_DIR=${PWD}/vlfeat/bin/glnxa64"
	@echo "export LD_LIBRARY_PATH=${PWD}/vlfeat/bin/glnxa64:$$""LD_LIBRARY_PATH"

.PHONY: otb-remote-module
otb-remote-module: ## install remote module otb
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_OTB}" ] || ( echo ">> OTB not found"; exit 1 )
	@[ "${OTB_APPLICATION_PATH}" ] || ( echo ">> OTB_APPLICATION_PATH is not set"; exit 1 )
	@mkdir -p build
	@cd build && cmake -DCMAKE_INSTALL_PREFIX=${CARS_VENV} -DOTB_BUILD_MODULE_AS_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release ../otb_remote_module && make install

.PHONY: install-deps
install-deps: venv ## install python libs
	@[ "${CHECK_NUMPY}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade "cython<3.0.0" numpy
	@[ "${CHECK_TBB}" ] ||${CARS_VENV}/bin/python -m pip install tbb==$(TBB_VERSION_SETUP)
	@[ "${CHECK_NUMBA}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade numba
	@[ "${CHECK_CYVLFEAT}" ] ||CFLAGS="-I${VLFEAT_INCLUDE_DIR}" LDFLAGS="-L${VLFEAT_LIBRARY_DIR}" ${CARS_VENV}/bin/python -m pip install --no-binary cyvlfeat cyvlfeat

.PHONY: install-deps-gdal
install-deps-gdal: install-deps ## create an healthy python environment for OTB / GDAL
	@[ "${CHECK_FIONA}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary fiona fiona
	@[ "${CHECK_RASTERIO}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary rasterio rasterio

.PHONY: install
install: install-deps-gdal otb-remote-module ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"

.PHONY: install-pandora-mccnn
install-pandora-mccnn: install-deps-gdal otb-remote-module  ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook,pandora_mccnn]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"

.PHONY: install-dev
install-dev: install-deps-gdal otb-remote-module ## install cars in dev editable mode (pip install -e .)
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install -e .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"

.PHONY: install-dev-otb-free
install-dev-otb-free: install-deps ## install cars in dev editable mode (pip install -e .) without recompiling otb remote modules, rasterio, fiona
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install -e .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; cars -h"

## Test section

.PHONY: test
test: ## run all tests + coverage html
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -o log_cli=true -o log_cli_level=${LOGLEVEL} --cov-config=.coveragerc --cov-report html --cov

.PHONY: test-ci
test-ci: ## run unit and pbs tests + coverage for cars-ci
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "unit_tests or pbs_cluster_tests" --durations=0 --log-date-format="%Y-%m-%d %H:%M:%S" --log-format="%(asctime)s [%(levelname)8s] (%(filename)s:%(lineno)s) : %(message)s"  -o log_cli=true -o log_cli_level=${LOGLEVEL} --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-end2end
test-end2end: ## run end2end tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "end2end_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-unit
test-unit: ## run unit tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "unit_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-pbs-cluster
test-pbs-cluster: ## run pbs cluster tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-slurm-cluster
test-slurm-cluster: ## run slurm cluster tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "slurm_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-notebook
test-notebook: ## run notebook tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "notebook_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

## Code quality, linting section

### Format with isort and black

.PHONY: format
format: format/isort format/black  ## run black and isort formatting (depends install)

.PHONY: format/isort
format/isort: ## run isort formatting (depends install)
	@echo "+ $@"
	@${CARS_VENV}/bin/isort cars tests

.PHONY: format/black
format/black: ## run black formatting (depends install)
	@echo "+ $@"
	@${CARS_VENV}/bin/black cars tests

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint: lint/isort lint/black lint/flake8 lint/pylint ## check code quality and linting

.PHONY: lint/isort
lint/isort: ## check imports style with isort
	@echo "+ $@"
	@${CARS_VENV}/bin/isort --check cars tests

.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${CARS_VENV}/bin/black --check cars tests

.PHONY: lint/flake8
lint/flake8: ## check linting with flake8
	@echo "+ $@"
	@${CARS_VENV}/bin/flake8 cars tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${CARS_VENV}/bin/pylint cars tests --rcfile=.pylintrc --output-format=parseable | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

## Documentation section

.PHONY: docs
docs: ## build sphinx documentation
	@${CARS_VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${CARS_VENV}/bin/sphinx-build -M html docs/source/ docs/build -W --keep-going

## Notebook section

.PHONY: notebook
notebook: ## install Jupyter notebook kernel with venv and cars install
	@echo "Install Jupyter Kernel in virtualenv dir"
	@${CARS_VENV}/bin/python -m ipykernel install --sys-prefix --name=cars-${CARS_VERSION_MIN} --display-name=cars-${CARS_VERSION_MIN}
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After CARS virtualenv activation, please use following command to launch local jupyter notebook to open CARS Notebooks:"
	@echo "jupyter notebook"


# Dev section

.PHONY: dev
dev: install-dev docs notebook ## install CARS in dev mode : install-dev, notebook and docs

## Docker section

.PHONY: docker-deps
docker-deps: ## Check and build docker image cnes/cars-deps
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@docker pull hadolint/hadolint
	@echo "Check Dockerfile.deps with hadolint"
	@docker run --rm -i hadolint/hadolint < Dockerfile.deps
	@echo "Build Docker deps image CARS ${CARS_VERSION_MIN}"
# Set docker options like --build-arg
ifndef DOCKER_OPTIONS
	@docker build -t cnes/cars-deps:${CARS_VERSION_MIN} -t cnes/cars-deps:latest . -f Dockerfile.deps
else
	@docker build ${DOCKER_OPTIONS} -t cnes/cars-deps:${CARS_VERSION_MIN} -t cnes/cars-deps:latest . -f Dockerfile.deps
endif

.PHONY: docker
docker: docker-deps ## Check and build docker image cnes/cars (depending on cnes/cars-deps)
	@echo "Check Dockerfile with hadolint"
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker main image CARS ${CARS_VERSION_MIN}"
# Set docker options like --build-arg
ifndef DOCKER_OPTIONS
	@docker build -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest . -f Dockerfile
else
	@docker build ${DOCKER_OPTIONS} -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest . -f Dockerfile
endif

.PHONY: docker-jupyter
docker-jupyter: ## Check and build docker image cnes/cars-jupyter
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@docker pull hadolint/hadolint
	@echo "Check Dockerfile.jupyter with hadolint"
	@docker run --rm -i hadolint/hadolint < Dockerfile.jupyter
	@echo "Build Docker jupyter notebook image from CARS"
# Set docker options like --build-arg
ifndef DOCKER_OPTIONS
	@docker build -t cnes/cars-jupyter:$(CARS_VERSION_MIN) -t cnes/cars-jupyter:latest . -f Dockerfile.jupyter
else
	@docker build ${DOCKER_OPTIONS} -t cnes/cars-jupyter:$(CARS_VERSION_MIN) -t cnes/cars-jupyter:latest . -f Dockerfile.jupyter
endif
	@echo "Build Docker jupyter notebook image from CARS"

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-vlfeat clean-precommit clean-pyc clean-test clean-docs clean-notebook clean-dask ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${CARS_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit:
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc:
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f pylint-report.txt
	@rm -f debug.log

.PHONY: clean-docs
clean-docs:
	@echo "+ $@"
	@rm -rf docs/build/
	@rm -rf docs/source/api_reference/

.PHONY: clean-notebook
clean-notebook:
	@echo "+ $@"
	@find . -type d -name ".ipynb_checkpoints" -exec rm -fr {} +

.PHONY: clean-dask
clean-dask:
	@echo "+ $@"
	@find . -type d -name "dask-worker-space" -exec rm -fr {} +


.PHONY: clean-docker
clean-docker: ## clean docker image
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Clean Docker images cars-deps, cars, cars-jupyter ${CARS_VERSION_MIN}"
	@docker image rm -f cnes/cars-deps:${CARS_VERSION_MIN}
	@docker image rm -f cnes/cars-deps:latest
	@docker image rm -f cnes/cars:${CARS_VERSION_MIN}
	@docker image rm -f cnes/cars:latest
	@docker image rm -f cnes/cars-jupyter:${CARS_VERSION_MIN}
	@docker image rm -f cnes/cars-jupyter:latest

.PHONY: clean-vlfeat
clean-vlfeat:
	@echo "+ $@"
	@rm -rf vlfeat

.PHONY: profile-memory-report
profile-memory-report: ## build report after execution of cars with profiling memray mode (report biggest  memory occupation for each application), indicate the output_result directory file
	@for file in $(wildcard ./$(filter-out $@,$(MAKECMDGOALS))/profiling/memray/*.bin); do echo $$file && ${CARS_VENV}/bin/memray tree -b 10 $$file; done;

.PHONY: profile-memory-all
profile-memory-all: ## memory profiling at master orchestrator level (not only at worker level) with cars CLI command, uses config.json as input (please use sequential orchestrator mode and desactive profiling)
	@${CARS_VENV}/bin/memray run -o memray.result.bin ${CARS_VENV}/bin/cars $(wildcard ./$(filter-out $@,$(MAKECMDGOALS)))
	@${CARS_VENV}/bin/memray tree -b 50 memray.result.bin

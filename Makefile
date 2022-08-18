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
# Set pytest LOGLEVEL if not defined in command line
# Example: LOGLEVEL="DEBUG" make test
ifndef LOGLEVEL
	LOGLEVEL = "INFO"
endif

# Check CMAKE, OTB, GDAL variables before venv creation
CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_OTB = $(shell command -v otbcli_ReadImageInfo 2> /dev/null)
GDAL_VERSION = $(shell gdal-config --version)

# Check python install in VENV
CHECK_NUMPY = $(shell ${CARS_VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${CARS_VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${CARS_VENV}/bin/python -m pip list|grep rasterio)
CHECK_PYGDAL = $(shell ${CARS_VENV}/bin/python -m pip list|grep pygdal)
CHECK_TBB = $(shell ${CARS_VENV}/bin/python -m pip list|grep tbb)
CHECK_NUMBA = $(shell ${CARS_VENV}/bin/python -m pip list|grep numba)
TBB_VERSION_SETUP = $(shell cat setup.cfg | grep tbb |cut -d = -f 3 | cut -d ' ' -f 1)

# Check Docker
CHECK_DOCKER = $(shell docker -v)

# CARS version from setup.py
CARS_VERSION = $(shell python3 setup.py --version)
CARS_VERSION_MIN =$(shell echo ${CARS_VERSION} | cut -d . -f 1,2,3)

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      CARS MAKE HELP  LOGLEVEL=${LOGLEVEL}"
	@echo "  Dependencies: Install OTB and VLFEAT before !"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: check
check: ## check if cmake, OTB, VLFEAT, GDAL is installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_OTB}" ] || ( echo ">> OTB not found"; exit 1 )
	@[ "${OTB_APPLICATION_PATH}" ] || ( echo ">> OTB_APPLICATION_PATH is not set"; exit 1 )
	@[ "${GDAL_VERSION}" ] || ( echo ">> GDAL_VERSION is not set"; exit 1 )
	@[ "${VLFEAT_INCLUDE_DIR}" ] || ( echo ">> VLFEAT_INCLUDE_DIR is not set"; exit 1 )

.PHONY: venv
venv: check ## create virtualenv in CARS_VENV directory if not exists
	@test -d ${CARS_VENV} || python3 -m venv ${CARS_VENV}
	@${CARS_VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${CARS_VENV}/bin/activate

.PHONY: install-deps
install-deps: venv
	@[ "${CHECK_NUMPY}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade cython numpy
	@[ "${CHECK_FIONA}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary fiona fiona
	@[ "${CHECK_RASTERIO}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary rasterio rasterio
	@[ "${CHECK_PYGDAL}" ] ||${CARS_VENV}/bin/python -m pip install pygdal==$(GDAL_VERSION).*
	@[ "${CHECK_TBB}" ] ||${CARS_VENV}/bin/python -m pip install tbb==$(TBB_VERSION_SETUP)
	@[ "${CHECK_NUMBA}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade numba

.PHONY: install
install: install-deps  ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage : source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"

install-pandora-mccnn: install-deps  ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook,pandora_mccnn]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage : source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"


.PHONY: install-dev
install-dev: install-deps ## install cars in dev editable mode (pip install -e .)
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install -e .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage : source ${CARS_VENV}/bin/activate; source ${CARS_VENV}/bin/env_cars.sh; cars -h"

## Test section

.PHONY: test
test: install ## run all tests + coverage html
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -o log_cli=true -o log_cli_level=${LOGLEVEL} --cov-config=.coveragerc --cov-report html --cov

.PHONY: test-ci
test-ci: install ## run unit and pbs tests + coverage for cars-ci
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "unit_tests or pbs_cluster_tests" --durations=0 --log-date-format="%Y-%m-%d %H:%M:%S" --log-format="%(asctime)s [%(levelname)8s] (%(filename)s:%(lineno)s) : %(message)s"  -o log_cli=true -o log_cli_level=${LOGLEVEL} --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-end2end
test-end2end: install ## run end2end tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "end2end_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-unit
test-unit: install ## run unit tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "unit_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-pbs-cluster
test-pbs-cluster: install ## run pbs cluster tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-notebook
test-notebook: install-pandora-mccnn ## run notebook tests only
	@echo "Please source ${CARS_VENV}/bin/env_cars.sh before launching tests\n"
	@${CARS_VENV}/bin/pytest -m "notebook_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

## Code quality, linting section

### Format with isort and black

.PHONY: format
format: install format/isort format/black  ## run black and isort formatting (depends install)

.PHONY: format/isort
format/isort: install  ## run isort formatting (depends install)
	@echo "+ $@"
	@${CARS_VENV}/bin/isort cars tests

.PHONY: format/black
format/black: install  ## run black formatting (depends install)
	@echo "+ $@"
	@${CARS_VENV}/bin/black cars tests

### Check code quality and linting : isort, black, flake8, pylint

.PHONY: lint
lint: install lint/isort lint/black lint/flake8 lint/pylint ## check code quality and linting

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
docs: install ## build sphinx documentation
	@${CARS_VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${CARS_VENV}/bin/sphinx-build -M html docs/source/ docs/build

## Notebook section

.PHONY: notebook
notebook: install ## install Jupyter notebook kernel with venv and cars install
	@echo "Install Jupyter Kernel in virtualenv dir"
	@${CARS_VENV}/bin/python -m ipykernel install --sys-prefix --name=cars-${CARS_VERSION_MIN} --display-name=cars-${CARS_VERSION_MIN}
	@echo "Use jupyter kernelspec list to know where is the kernel"
	@echo " --> After CARS virtualenv activation, please use following command to launch local jupyter notebook to open CARS Notebooks:"
	@echo "jupyter notebook"


# Dev section

.PHONY: dev
dev: install-dev docs notebook ## Install CARS in dev mode : install-dev, notebook and docs

## Docker section

.PHONY: docker
docker: ## Check and build docker images (cnes/cars and cnes/cars-jupyter)
	@@[ "${CHECK_DOCKER}" ] || ( echo ">> docker not found"; exit 1 )
	@echo "Check Dockerfiles with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@docker run --rm -i hadolint/hadolint < Dockerfile.jupyter
	@docker run --rm -i hadolint/hadolint < Dockerfile.tutorial
	@echo "Build Docker main image CARS ${CARS_VERSION_MIN}"
	@docker build -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest . -f Dockerfile
	@echo "Build Docker jupyter notebook image from CARS"
	@docker build -t cnes/cars-jupyter:${CARS_VERSION_MIN} -t cnes/cars-jupyter:latest . -f Dockerfile.jupyter
	@echo "Build Docker jupyter tutorial notebook image from CARS"
	@docker build -t cnes/cars-tutorial:${CARS_VERSION_MIN} -t cnes/cars-tutorial:latest . -f Dockerfile.tutorial
	
	## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-docs clean-notebook clean-dask ## remove all build, test, coverage and Python artifacts

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
	@echo "Clean Docker image cnes/cars ${CARS_VERSION_MIN}"
	@docker image rm cnes/cars:${CARS_VERSION_MIN}
	@docker image rm cnes/cars:latest

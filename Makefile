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

# Check python install in VENV
CHECK_NUMPY = $(shell ${CARS_VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${CARS_VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${CARS_VENV}/bin/python -m pip list|grep rasterio)
CHECK_TBB = $(shell ${CARS_VENV}/bin/python -m pip list|grep tbb)
CHECK_NUMBA = $(shell ${CARS_VENV}/bin/python -m pip list|grep numba)
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
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: venv
venv: ## create virtualenv in CARS_VENV directory if not exists
	@test -d ${CARS_VENV} || python3 -m venv ${CARS_VENV}
	@${CARS_VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${CARS_VENV}/bin/activate

.PHONY: install-deps
install-deps: venv ## install python libs
	@[ "${CHECK_NUMPY}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade cython numpy
	@[ "${CHECK_TBB}" ] ||${CARS_VENV}/bin/python -m pip install tbb==$(TBB_VERSION_SETUP)
	@[ "${CHECK_NUMBA}" ] ||${CARS_VENV}/bin/python -m pip install --upgrade numba

.PHONY: install-deps-gdal
install-deps-gdal: install-deps ## create an healthy python environment for GDAL/ proj
	@[ "${CHECK_FIONA}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary fiona fiona
	@[ "${CHECK_RASTERIO}" ] ||${CARS_VENV}/bin/python -m pip install --no-binary rasterio rasterio

.PHONY: install
install: install-deps ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; cars -h"

.PHONY: install-gdal
install-gdal: install-deps-gdal ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook,pandora_mccnn]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; cars -h"

.PHONY: install-pandora-mccnn
install-pandora-mccnn: install-deps  ## install cars (not editable) with dev, docs, notebook dependencies
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install .[dev,docs,notebook,pandora_mccnn]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in virtualenv ${CARS_VENV}"

.PHONY: install-dev
install-dev: install-deps ## install cars in dev editable mode (pip install -e .) without recompiling rasterio, fiona
	@test -f ${CARS_VENV}/bin/cars || ${CARS_VENV}/bin/pip install -e .[dev,docs,notebook]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${CARS_VENV}/bin/pre-commit install -t pre-commit
	@test -f .git/hooks/pre-push || ${CARS_VENV}/bin/pre-commit install -t pre-push
	@echo "CARS ${CARS_VERSION} installed in dev mode in virtualenv ${CARS_VENV}"
	@echo "CARS venv usage: source ${CARS_VENV}/bin/activate; cars -h"

## Test section

.PHONY: test
test: ## run unit tests without SLURM cluster + coverage html
	@${CARS_VENV}/bin/pytest -m "unit_tests and not pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL} --cov-config=.coveragerc --cov-report html --cov

.PHONY: test-ci
test-ci: ## run unit and pbs tests + coverage for cars-ci
	@${CARS_VENV}/bin/pytest -m "unit_tests or pbs_cluster_tests" --durations=0 --log-date-format="%Y-%m-%d %H:%M:%S" --log-format="%(asctime)s [%(levelname)8s] (%(filename)s:%(lineno)s) : %(message)s"  -o log_cli=true -o log_cli_level=${LOGLEVEL} --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

.PHONY: test-end2end
test-end2end: ## run end2end tests only
	@${CARS_VENV}/bin/pytest -m "end2end_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-unit
test-unit: ## run unit tests only
	@${CARS_VENV}/bin/pytest -m "unit_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-pbs-cluster
test-pbs-cluster: ## run pbs cluster tests only
	@${CARS_VENV}/bin/pytest -m "pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-slurm-cluster
test-slurm-cluster: ## run slurm cluster tests only
	@${CARS_VENV}/bin/pytest -m "slurm_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

.PHONY: test-notebook
test-notebook: ## run notebook tests only
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
	@echo "Check Dockerfile with hadolint"
	@docker run --rm -i hadolint/hadolint < Dockerfile

.PHONY: docker
docker: docker-deps ## Check and build docker image cnes/cars 
	@echo "Build Docker main image CARS ${CARS_VERSION_MIN}"
# Set docker options like --build-arg
ifndef DOCKER_OPTIONS
	@docker build -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest . -f Dockerfile
else
	@docker build ${DOCKER_OPTIONS} -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest . -f Dockerfile
endif


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
	@echo "Clean Docker images cars ${CARS_VERSION_MIN}"
	@docker image rm -f cnes/cars:${CARS_VERSION_MIN}
	@docker image rm -f cnes/cars:latest


.PHONY: profile-memory-report
profile-memory-report: ## build report after execution of cars with profiling memray mode (report biggest  memory occupation for each application), indicate the output_result directory file
	@for file in $(wildcard ./$(filter-out $@,$(MAKECMDGOALS))/profiling/memray/*.bin); do echo $$file && ${CARS_VENV}/bin/memray tree -b 10 $$file; done;

.PHONY: profile-memory-all
profile-memory-all: ## memory profiling at master orchestrator level (not only at worker level) with cars CLI command, uses config.json as input (please use sequential orchestrator mode and desactive profiling)
	@${CARS_VENV}/bin/memray run -o memray.result.bin ${CARS_VENV}/bin/cars $(wildcard ./$(filter-out $@,$(MAKECMDGOALS)))
	@${CARS_VENV}/bin/memray tree -b 50 memray.result.bin

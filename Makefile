# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

# GLOBAL VARIABLES
# Set Virtualenv directory name
VENV = "venv"

CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_OTB = $(shell command -v otbcli_ReadImageInfo 2> /dev/null)

CHECK_NUMPY = $(shell ${VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${VENV}/bin/python -m pip list|grep rasterio)
CHECK_PYGDAL = $(shell ${VENV}/bin/python -m pip list|grep pygdal)

GDAL_VERSION = $(shell gdal-config --version)
CARS_VERSION = $(shell python3 setup.py --version)
CARS_VERSION_MIN =$(shell echo ${CARS_VERSION} | cut -d . -f 1,2,3)

# TARGETS
.PHONY: help venv install test lint format docs docker clean

help: ## this help
	@echo "      CARS MAKE HELP"
	@echo "  Dependencies: Install OTB and VLFEAT before !\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

check: ## check if cmake, OTB, VLFEAT, GDAL is installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_OTB}" ] || ( echo ">> OTB not found"; exit 1 )
	@[ "${OTB_APPLICATION_PATH}" ] || ( echo ">> OTB_APPLICATION_PATH is not set"; exit 1 )
	@[ "${GDAL_DATA}" ] || ( echo ">> GDAL_DATA is not set"; exit 1 )
	@[ "${GDAL_VERSION}" ] || ( echo ">> GDAL_VERSION is not set"; exit 1 )
	@[ "${VLFEAT_INCLUDE_DIR}" ] || ( echo ">> VLFEAT_INCLUDE_DIR is not set"; exit 1 )

venv: check ## create virtualenv in "venv" dir if not exists
	@test -d ${VENV} || virtualenv -p `which python3` ${VENV}
	@${VENV}/bin/python -m pip install --upgrade pip setuptools # no check to upgrade each time
	@touch ${VENV}/bin/activate

install-deps: venv
	@[ "${CHECK_NUMPY}" ] ||${VENV}/bin/python -m pip install --upgrade cython numpy
	@[ "${CHECK_FIONA}" ] ||${VENV}/bin/python -m pip install --no-binary fiona fiona
	@[ "${CHECK_RASTERIO}" ] ||${VENV}/bin/python -m pip install --no-binary rasterio rasterio
	@[ "${CHECK_PYGDAL}" ] ||${VENV}/bin/python -m pip install pygdal==$(GDAL_VERSION).*

install: install-deps  ## install and set env
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose .
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

install-dev: install-deps ## install cars in dev mode and set env
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose -e .[dev]
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed in dev mode"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

test: install-dev ## run all tests + coverage html
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -o log_cli=true -o log_cli_level=INFO --cov-config=.coveragerc --cov-report html --cov

test-ci: install-dev ## run unit and pbs tests + coverage for cars-ci
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "unit_tests or pbs_cluster_tests" -o log_cli=true -o log_cli_level=INFO --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

test-end2end: install-dev ## run end2end tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "end2end_tests" -o log_cli=true -o log_cli_level=INFO

test-unit: install-dev ## run unit tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "unit_tests" -o log_cli=true -o log_cli_level=INFO

test-pbs-cluster: install-dev ## run pbs cluster tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "pbs_cluster_tests" -o log_cli=true -o log_cli_level=INFO

test-notebook: install-dev ## run notebook tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "notebook_tests" -o log_cli=true -o log_cli_level=INFO

lint-ci: install-dev ## run lint tools for cars-ci
	@${VENV}/bin/isort --check cars tests
	@${VENV}/bin/black --check cars tests
	@${VENV}/bin/flake8 cars tests
	@${VENV}/bin/pylint cars tests --rcfile=.pylintrc --output-format=parseable > pylint-report.txt || cat pylint-report.txt

lint: install-dev  ## run lint tools (depends install)
	@${VENV}/bin/isort --check cars tests
	@${VENV}/bin/black --check cars tests
	@${VENV}/bin/flake8 cars tests
	@${VENV}/bin/pylint cars tests --rcfile=.pylintrc

format: install-dev  ## run black and isort (depends install)
	@${VENV}/bin/isort cars tests
	@${VENV}/bin/black cars tests

docs:  ## build sphinx documentation (requires doc venv TODO)
	@cd docs/ && make clean && make html && cd ..

docker: ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image CARS ${CARS_VERSION_MIN}"
	@docker build -t cnes/cars:${CARS_VERSION_MIN} .

clean: ## clean: remove venv, cars build, cache, ...
	@rm -rf ${VENV}
	@rm -rf dist
	@rm -rf build
	@rm -rf cars.egg-info
	@rm -rf **/__pycache__
	@rm -rf .eggs
	@rm -rf dask-worker-space/
	@rm -f .coverage
	@rm -rf .coverage.*

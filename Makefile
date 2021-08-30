# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

# GLOBAL VARIABLES
# Set Virtualenv directory name
VENV = "venv"
# Set LOGLEVEL if not defined in command line
# Example: LOGLEVEL="DEBUG" make help
ifndef LOGLEVEL
	LOGLEVEL = "INFO"
endif

CHECK_CMAKE = $(shell command -v cmake 2> /dev/null)
CHECK_OTB = $(shell command -v otbcli_ReadImageInfo 2> /dev/null)

CHECK_NUMPY = $(shell ${VENV}/bin/python -m pip list|grep numpy)
CHECK_FIONA = $(shell ${VENV}/bin/python -m pip list|grep Fiona)
CHECK_RASTERIO = $(shell ${VENV}/bin/python -m pip list|grep rasterio)
CHECK_PYGDAL = $(shell ${VENV}/bin/python -m pip list|grep pygdal)
CHECK_TBB = $(shell ${VENV}/bin/python -m pip list|grep tbb)
CHECK_NUMBA = $(shell ${VENV}/bin/python -m pip list|grep numba)

TBB_VERSION_SETUP = $(shell cat setup.cfg | grep tbb |cut -d = -f 3 | cut -d ' ' -f 1)

GDAL_VERSION = $(shell gdal-config --version)
CARS_VERSION = $(shell python3 setup.py --version)
CARS_VERSION_MIN =$(shell echo ${CARS_VERSION} | cut -d . -f 1,2,3)

# TARGETS
.PHONY: help check venv install-deps install install-notebook install-doc install-dev test test-ci test-end2end test-unit test-pbs-cluster test-notebook lint format doc notebook docker clean

help: ## this help
	@echo "      CARS MAKE HELP  LOGLEVEL=${LOGLEVEL}"
	@echo "  Dependencies: Install OTB and VLFEAT before !\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

check: ## check if cmake, OTB, VLFEAT, GDAL is installed
	@[ "${CHECK_CMAKE}" ] || ( echo ">> cmake not found"; exit 1 )
	@[ "${CHECK_OTB}" ] || ( echo ">> OTB not found"; exit 1 )
	@[ "${OTB_APPLICATION_PATH}" ] || ( echo ">> OTB_APPLICATION_PATH is not set"; exit 1 )
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
	@[ "${CHECK_TBB}" ] ||${VENV}/bin/python -m pip install tbb==$(TBB_VERSION_SETUP)
	@[ "${CHECK_NUMBA}" ] ||${VENV}/bin/python -m pip install --upgrade numba

install: install-deps  ## install cars
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose .
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

install-notebook: install-deps  ## install cars with Jupyter notebooks packages
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose .[notebook]
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed with Jupyter notebook packages"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

install-doc: install-deps  ## install cars with Sphinx documentation dependencies
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose .[doc]
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed with Sphinx docs dependencies"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

install-dev: install-deps ## install cars in dev mode
	@test -f ${VENV}/bin/cars || ${VENV}/bin/pip install --verbose -e .[dev]
	@echo "\n --> CARS installed in virtualenv ${VENV}"
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${VENV}/bin/pre-commit install -t pre-commit
	@chmod +x ${VENV}/bin/register-python-argcomplete
	@echo "CARS ${CARS_VERSION} installed in dev mode"
	@echo "CARS venv usage : source ${VENV}/bin/activate; source ${VENV}/bin/env_cars.sh; cars -h"

test: install-dev ## run all tests + coverage html
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -o log_cli=true -o log_cli_level=${LOGLEVEL} --cov-config=.coveragerc --cov-report html --cov

test-ci: install-dev ## run unit and pbs tests + coverage for cars-ci
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "unit_tests or pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL} --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

test-end2end: install-dev ## run end2end tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "end2end_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

test-unit: install-dev ## run unit tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "unit_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

test-pbs-cluster: install-dev ## run pbs cluster tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "pbs_cluster_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

test-notebook: install-dev ## run notebook tests only
	@echo "Please source ${VENV}/bin/env_cars.sh before launching tests\n"
	@${VENV}/bin/pytest -m "notebook_tests" -o log_cli=true -o log_cli_level=${LOGLEVEL}

lint: install-dev ## run lint tools (depends install-dev)
	@echo "CARS linting isort check"
	@${VENV}/bin/isort --check cars tests
	@echo "CARS linting black check"
	@${VENV}/bin/black --check cars tests
	@echo "CARS linting flake8 check"
	@${VENV}/bin/flake8 cars tests
	@echo "CARS linting pylint check"
	@${VENV}/bin/pylint cars tests --rcfile=.pylintrc --output-format=parseable > pylint-report.txt 
	@cat pylint-report.txt

format: install-dev  ## run black and isort formatting (depends install-dev)
	@${VENV}/bin/isort cars tests
	@${VENV}/bin/black cars tests

doc: install-doc ## build sphinx documentation
	@${VENV}/bin/sphinx-build -M clean docs/source/ docs/build
	@${VENV}/bin/sphinx-apidoc -o docs/source/apidoc/ cars
	@${VENV}/bin/sphinx-build -M html docs/source/ docs/build

notebook: install-notebook ## Install Jupyter notebook kernel with venv and cars install
	@echo "\nInstall Jupyter Kernel and launch Jupyter notebooks environment"
	@${VENV}/bin/python -m ipykernel install --sys-prefix --name=cars-$(VENV) --display-name=cars-$(CARS_VERSION)
	@echo "\n --> After CARS virtualenv activation, please use following command to launch local jupyter notebook to open CARS Notebooks:"
	@echo "jupyter notebook"

docker: ## Build docker image (and check Dockerfile)
	@echo "Check Dockerfile with hadolint"
	@docker pull hadolint/hadolint
	@docker run --rm -i hadolint/hadolint < Dockerfile
	@echo "Build Docker image CARS ${CARS_VERSION_MIN}"
	@docker build -t cnes/cars:${CARS_VERSION_MIN} -t cnes/cars:latest .

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

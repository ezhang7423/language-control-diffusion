.ONESHELL:

#* Variables
SHELL := /usr/bin/bash
PYTHON := python
PYTHONPATH := `pwd`
CONDA != type -P mamba >/dev/null 2>&1 && echo "mamba" || echo "conda"
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ACTIVATE_2 = source $$(conda info --base)/etc/profile.d/$(CONDA).sh ; $(CONDA) activate ; $(CONDA) activate # hack to make work with mamba


#* Docker variables
IMAGE := lcd
VERSION := latest

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	@set -e
	@export OG_DIR="$$(pwd -P)"

	git -c submodule."submodules/hulc-data".update=none submodule update --init --recursive;
	@if [ -z "$(NO_DATA)" ]; then\
		echo "Downloading data...";\
		git submodule update --init --recursive ./submodules/hulc-data;\
	fi
	! type -P poetry &> /dev/null && curl -sSL https://install.python-poetry.org | python3 -
	! type -P $(CONDA) &> /dev/null && { echo "Please install mamba to continue (https://mamba.readthedocs.io/en/latest/installation.html)"; exit 1; }

	# install lcd conda environment
	$(CONDA) create -n lcd python=3.8 -y
	$(CONDA_ACTIVATE) lcd
	$(CONDA_ACTIVATE_2) lcd

	type python
	
	pip install setuptools==57.5.0
	
	$(CONDA) install -c fvcore -c iopath -c conda-forge fvcore iopath -y
	$(CONDA) install pytorch3d=0.7.2 -c pytorch3d -y

	# install hulc-baseline
	cd ./submodules/hulc-baseline
	. install.sh

	# install poetry environment
	cd $${OG_DIR}
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n


.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus ./src/**/*.py
	poetry run isort --settings-path pyproject.toml ./src
	poetry run black --config pyproject.toml ./src

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=lcd tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./src
	poetry run black --diff --check --config pyproject.toml ./src
	poetry run darglint --verbosity 2 lcd tests

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./src

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive lcd tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

#* Docker
# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove

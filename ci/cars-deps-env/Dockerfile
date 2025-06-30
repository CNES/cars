ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL

WORKDIR /app/cars

# The Dockerfile only use pyproject.toml to build the environment image
COPY pyproject.toml .

COPY certs certs

# for pip
ENV PIP_CERT="/app/cars/certs/ca-bundle.crt"

# for uv
ENV SSL_CERT_FILE="/app/cars/certs/ca-bundle.crt"
ENV UV_DEFAULT_INDEX="${PIP_INDEX_URL}"

# install uv which will then be used to find and install dependencies
RUN pip install uv

# Generate a requirements.txt file from pyproject.toml containing cars dependencies
# without having to compile and install cars
RUN uv pip compile pyproject.toml --extra dev \
                                  --extra notebook \
                                  --extra docs \
                                  --output-file requirements.txt

# uv variable defining default env
ENV VIRTUAL_ENV=/app/cars/venv

# Install cars requirements
RUN uv venv ${VIRTUAL_ENV} && uv pip install --no-cache meson-python meson ninja setuptools_scm setuptools wheel pybind11 -r requirements.txt
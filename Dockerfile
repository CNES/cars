# hadolint ignore=DL3007
FROM orfeotoolbox/otb:9.1.1_ubuntu24
LABEL maintainer="CNES"

# Dependencies packages
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


# Python 3.10
# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y --quiet git python3.10-dev python3.10-venv \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install dependancies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3013
RUN pip uninstall numpy -y  \
    && pip install numpy \
    && pip install --no-cache-dir --no-binary fiona fiona \
    && pip install --no-cache-dir --no-binary rasterio rasterio

# Install cars: from source or from pypi if version
ARG version
# hadolint ignore=DL3003,DL3013
RUN if [ -z "$version" ] ; then git clone --depth 1 https://github.com/CNES/cars.git && cd cars && python3 -m pip install --no-cache-dir build && python3 -m build &&  pip install --no-cache-dir dist/*.whl && cd - && rm -rf cars; \
    else pip install --no-cache-dir cars==$version; fi

# Launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

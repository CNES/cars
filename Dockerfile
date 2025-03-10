# hadolint ignore=DL3007
FROM orfeotoolbox/otb:latest
LABEL maintainer="CNES"

# Dependencies packages
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa && rm -rf /var/lib/apt/lists/*

# Python 3.10
# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y --quiet git python3.10-dev \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install dependancies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3013
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \ 
    && python3 -m pip install --no-cache-dir --upgrade requests  \
    && python3 -m pip install --no-cache-dir --no-binary fiona fiona \
    && python3 -m pip install --no-cache-dir --no-binary rasterio rasterio

# Install cars: from source or from pypi if version
ARG version
# hadolint ignore=DL3003,DL3013
RUN if [ -z "$version" ] ; then git clone --depth 1 https://github.com/CNES/cars.git && cd cars && python3 -m pip install --no-cache-dir build && python3 -m build && python3 -m pip install --no-cache-dir dist/*.whl && cd - && rm -rf cars; \
    else python3 -m pip install --no-cache-dir cars==$version; fi

# Launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

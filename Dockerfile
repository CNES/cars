# hadolint ignore=DL3007
ARG REGISTRY=""

FROM  ${REGISTRY}orfeotoolbox/otb:9.1.1_ubuntu24
LABEL maintainer="CNES"

ARG USE_CERTS=false

COPY certs* /tmp/certs

RUN if [ "$USE_CERTS" = "true" ]; then \
      cp /tmp/certs/*.* /usr/local/share/ca-certificates/ ; \
      update-ca-certificates ; \
    else \
      echo "Skipping cert installation" ; \
    fi

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

ENV  PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# hadolint ignore=DL3013
RUN pip uninstall numpy -y  \
    && pip install numpy

ENV CONSTRAINTS_FILE=/opt/constraints.txt
RUN echo "rasterio --no-binary rasterio" > $CONSTRAINTS_FILE && \
    echo "fiona --no-binary fiona" >> $CONSTRAINTS_FILE

# Install cars: from source or from pypi if version
ARG version
ARG GIT_BRANCH=master
ARG CARS_URL=https://github.com/CNES/cars.git
# hadolint ignore=DL3003,DL3013
RUN if [ -z "$version" ] ; then git clone --single-branch --branch $GIT_BRANCH --depth 1 ${CARS_URL} --single && cd cars && python3 -m pip install --no-cache-dir -c $CONSTRAINTS_FILE build && python3 -m build &&  pip install --no-cache-dir -c $CONSTRAINTS_FILE dist/*.whl && cd - && rm -rf cars; \
    else pip install --no-cache-dir -c $CONSTRAINTS_FILE cars==$version; fi

# Launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

# hadolint ignore=DL3007
FROM orfeotoolbox/otb:latest
LABEL maintainer="CNES"

# Dependencies packages
# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y --quiet \
    git \
    libpython3.8 \
    python3.8-dev \
    python3.8-venv \
    python3.8 \
    python3-pip \
    python3-numpy \
    python3-virtualenv \
    && rm -rf /var/lib/apt/lists/*

# copy and install cars with mccnn plugin capabilities installed (but not configured by default)
WORKDIR /cars
COPY . /cars/

# Install fiona and rasterio with gdal / proj from otb
RUN make clean && make install-gdal

# source venv/bin/activate in docker mode
ENV VIRTUAL_ENV='/cars/venv'
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3013,SC2102
RUN python -m pip cache purge

# launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

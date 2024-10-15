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
WORKDIR /app


# Create a virtual environment
RUN python3 -m venv /app/venv

# source venv/bin/activate in docker mode
ENV VIRTUAL_ENV='/app/venv'
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Copy only necessary files for installation
COPY . /app/cars

# Install fiona and rasterio with gdal / proj from otb
WORKDIR /app/cars
RUN CARS_VENV=$VIRTUAL_ENV make clean && CARS_VENV=$VIRTUAL_ENV make install-gdal-dev


# hadolint ignore=DL3013,SC2102
RUN python -m pip cache purge

# launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

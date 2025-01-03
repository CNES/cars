# hadolint ignore=DL3007
FROM orfeotoolbox/otb:latest
LABEL maintainer="CNES"

# Dependencies packages
# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y --quiet \
    git \
    libpython3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10 \
    python3-pip \
    python3-numpy \
    python3-virtualenv \
    && rm -rf /var/lib/apt/lists/*

# copy and install cars with mccnn plugin capabilities installed (but not configured by default)
WORKDIR /app

# Create a virtual environment
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && python3 -m venv /app/venv

# source venv/bin/activate in docker mode
ENV VIRTUAL_ENV='/app/venv'

# Copy only necessary files for installation
COPY . /app/cars

# Install fiona and rasterio with gdal / proj from otb
WORKDIR /app/cars
RUN echo "CARS installation" && CARS_VENV=$VIRTUAL_ENV make clean && CARS_VENV=$VIRTUAL_ENV make install-gdal-dev
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# hadolint ignore=DL3013,SC2102
RUN python -m pip cache purge

# launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

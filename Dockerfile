FROM ubuntu:18.04
MAINTAINER David Youssefi <david.youssefi@cnes.fr>
ARG DEBIAN_FRONTEND=noninteractive

# standard packages
RUN apt-get update && apt-get install -y \
    cmake-curses-gui \
    git \
    libgl1 libglu1 libgl1-mesa-dev \
    libsm6 libxext6 libxrender-dev \
    python3 \
    python3-pip \
    python3-numpy \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install orfeo toolbox
WORKDIR /opt/otb
RUN wget -nv https://www.orfeo-toolbox.org/packages/OTB-6aca6c3e-Linux64.run \
    && chmod +x OTB-6aca6c3e-Linux64.run \
    && ./OTB-6aca6c3e-Linux64.run --target /usr/local/otb \
    && /bin/bash -c 'source /usr/local/otb/otbenv.profile' \
    && ctest -S /usr/local/otb/share/otb/swig/build_wrapping.cmake -VV \
    && rm OTB-6aca6c3e-Linux64.run
ENV LD_LIBRARY_PATH=/usr/local/otb/lib:$LD_LIBRARY_PATH \
    OTB_APPLICATION_PATH=/usr/local/otb/lib/otb/applications:$OTB_APPLICATION_PATH \
    PATH=/usr/local/otb/bin/:$PATH \
    PYTHONPATH=/usr/local/otb/lib/python:$PYTHONPATH \
    GDAL_DATA=/usr/local/otb/share/gdal \
    GEOTIFF_CSV=/usr/local/otb/share/epsg_csv
COPY gdal-config /usr/local/otb/bin/

# install vlfeat
RUN cd /tmp \
    && git clone https://github.com/vlfeat/vlfeat.git vlfeat \
    && cd vlfeat \
    && make \
    && cp bin/glnxa64/libvl.so /usr/local/lib \
    && mkdir -p /usr/local/include/vl \
    && cp vl/*.h /usr/local/include/vl \
    && cd /tmp \
    && rm -rf vlfeat

# copy
COPY . /cars/
WORKDIR /cars
COPY geoid /usr/local/geoid
ENV OTB_GEOID_FILE=/usr/local/geoid/egm96.grd

# install cars
RUN python3 -m pip install six setuptools --upgrade
RUN python3 -m pip install pygdal==$(gdal-config --version).*
ENV VLFEAT_INCLUDE_DIR=/usr/local/include
RUN python3 -m pip install /cars/.

RUN ln -s /usr/bin/python3 /usr/bin/python

# source /usr/local/bin/env_cars.sh
ENV CARSPATH=/cars \
    CARS_STATIC_CONFIGURATION=/usr/local \
    OTB_APPLICATION_PATH=/usr/lib:/usr/local/lib:$OTB_APPLICATION_PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    CARS_NB_WORKERS_PER_PBS_JOB=2 \
    OMP_NUM_THREADS=4 \
    OTB_MAX_RAM_HINT=2000 \
    OTB_LOGGER_LEVEL=WARNING \
    GDAL_CACHEMAX=128 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
    CARS_TEST_TEMPORARY_DIR=/tmp \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV NUMBA_NUM_THREADS=$OMP_NUM_THREADS \
    OPJ_NUM_THREADS=$OMP_NUM_THREADS \
    GDAL_NUM_THREADS=$OMP_NUM_THREADS

ENV OTB_MAX_RAM_HINT=1000

# launch cars
ENTRYPOINT ["cars_cli"]
CMD ["-h"]
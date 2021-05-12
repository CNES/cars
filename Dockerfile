FROM ubuntu:18.04
LABEL maintainer="CNES"

# Avoid apt install interactive questions.
ARG DEBIAN_FRONTEND=noninteractive

# Dependencies packages
RUN apt-get update && apt-get install --no-install-recommends -y --quiet \
    cmake-curses-gui=3.10.2-1ubuntu2.18.04.1 \
    git=1:2.17.1-1ubuntu0.8 \
    wget=1.19.4-1ubuntu2.2 \
    file=1:5.32-2ubuntu0.4 \
    apt-utils=1.6.12ubuntu0.2 \
    gcc=4:7.4.0-1ubuntu2.3 \
    g++=4:7.4.0-1ubuntu2.3 \
    make=4.1-9.1ubuntu1 \
    libpython3.6=3.6.9-1~18.04ubuntu1.4 \
    python3.6-dev=3.6.9-1~18.04ubuntu1.4 \
    libgl1=1.0.0-2ubuntu2.3 \
    libglu1-mesa=9.0.0-2.1build1 \
    libgl1-mesa-dev=20.0.8-0ubuntu1~18.04.1 \
    libsm6=2:1.2.2-1 \
    libxext6=2:1.3.3-1 \
    libxrender-dev=1:0.9.10-1 \
    python3=3.6.7-1~18.04\
    python3-pip=9.0.1-2.3~ubuntu1.18.04.4 \
    python3-numpy=1:1.13.3-2ubuntu1 \
    libgtk2.0-dev=2.24.32-1ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

# install orfeo toolbox
WORKDIR /opt/otb
RUN wget -nv https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-7.2.0-Linux64.run \
    && chmod +x OTB-7.2.0-Linux64.run \
    && ./OTB-7.2.0-Linux64.run --target /usr/local/otb \
    && /bin/bash -c 'source /usr/local/otb/otbenv.profile' \
    && ctest -S /usr/local/otb/share/otb/swig/build_wrapping.cmake -VV \
    && rm OTB-7.2.0-Linux64.run
ENV LD_LIBRARY_PATH=/usr/local/otb/lib:$LD_LIBRARY_PATH \
    OTB_APPLICATION_PATH=/usr/local/otb/lib/otb/applications:$OTB_APPLICATION_PATH \
    PATH=/usr/local/otb/bin/:$PATH \
    PYTHONPATH=/usr/local/otb/lib/python:$PYTHONPATH \
    GDAL_DATA=/usr/local/otb/share/gdal \
    PROJ_LIB=/usr/local/otb/share/proj \
    GEOTIFF_CSV=/usr/local/otb/share/epsg_csv
COPY gdal-config /usr/local/otb/bin/

# install vlfeat
WORKDIR /opt
RUN git clone https://github.com/vlfeat/vlfeat.git vlfeat

WORKDIR /opt/vlfeat
RUN make \
    && cp bin/glnxa64/libvl.so /usr/local/lib \
    && mkdir -p /usr/local/include/vl \
    && cp -r vl/*.h /usr/local/include/vl
WORKDIR /opt
RUN rm -rf vlfeat
ENV VLFEAT_INCLUDE_DIR=/usr/local/include


# copy cars
WORKDIR /cars
COPY . /cars/

# install cars
RUN python3 -m pip --no-cache-dir install pip setuptools cython --upgrade
RUN python3 -m pip --no-cache-dir install --no-binary fiona fiona
RUN python3 -m pip --no-cache-dir install --no-binary rasterio rasterio
RUN python3 -m pip --no-cache-dir install pygdal=="$(gdal-config --version).*"

RUN python3 -m pip --no-cache-dir install /cars/.

# source /usr/local/bin/env_cars.sh
ENV OTB_APPLICATION_PATH=/usr/lib:/usr/local/lib:$OTB_APPLICATION_PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    CARS_NB_WORKERS_PER_PBS_JOB=2 \
    OMP_NUM_THREADS=4 \
    OTB_MAX_RAM_HINT=2000 \
    OTB_LOGGER_LEVEL=WARNING \
    GDAL_CACHEMAX=128 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV NUMBA_NUM_THREADS=$OMP_NUM_THREADS \
    OPJ_NUM_THREADS=$OMP_NUM_THREADS \
    GDAL_NUM_THREADS=$OMP_NUM_THREADS

ENV OTB_MAX_RAM_HINT=1000

# launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

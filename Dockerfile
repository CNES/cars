FROM ubuntu:20.04
LABEL maintainer="CNES"

# Avoid apt install interactive questions.
ARG DEBIAN_FRONTEND=noninteractive

# Dependencies packages
RUN apt-get update && apt-get install --no-install-recommends -y --quiet \
    cmake-curses-gui=3.16.3-1ubuntu1 \
    git=1:2.25.1-1ubuntu3.2 \
    wget=1.20.3-1ubuntu1 \
    file=1:5.38-4 \
    apt-utils=2.0.6 \
    gcc=4:9.3.0-1ubuntu2 \
    g++=4:9.3.0-1ubuntu2 \
    make=4.2.1-1.2 \
    libpython3.8=3.8.10-0ubuntu1~20.04.2 \
    python3.8-dev=3.8.10-0ubuntu1~20.04.2 \
    python3.8-venv=3.8.10-0ubuntu1~20.04.2 \
    python3=3.8.2-0ubuntu2 \
    python3-pip=20.0.2-5ubuntu1.6 \
    python3-numpy=1:1.17.4-5ubuntu3 \
    unzip=6.0-25ubuntu1 \
    ninja-build=1.10.0-1build1 \
    libboost-date-time-dev=1.71.0.0ubuntu2 \
    libboost-filesystem-dev=1.71.0.0ubuntu2 \
    libboost-graph-dev=1.71.0.0ubuntu2 \
    libboost-program-options-dev=1.71.0.0ubuntu2 \
    libboost-system-dev=1.71.0.0ubuntu2 \
    libboost-thread-dev=1.71.0.0ubuntu2 \
    libgdal-dev=3.0.4+dfsg-1build3 \
    libinsighttoolkit4-dev=4.13.2-dfsg1-8 \
    libopenthreads-dev=3.6.4+dfsg1-3build2 \
    libossim-dev=2.9.1-2build1 \
    libtinyxml-dev=2.6.2-4build1 \
    libmuparser-dev=2.2.6.1+dfsg-1build1 \
    libmuparserx-dev=4.0.7+dfsg-3build1 \
    libsvm-dev=3.24+ds-3build1 \
    swig=4.0.1-5build1 \
    libfftw3-dev=3.3.8-2ubuntu1 \
    python3-virtualenv=20.0.17-1ubuntu0.4 \
    && rm -rf /var/lib/apt/lists/*

# install orfeo toolbox
WORKDIR /opt/otb
RUN wget -q https://www.orfeo-toolbox.org/packages/archives/OTB/OTB-7.4.0.zip -O /tmp/OTB-7.4.0.zip && \
    unzip -q /tmp/OTB-7.4.0.zip && rm /tmp/OTB-7.4.0.zip
WORKDIR /opt/otb/build
RUN cmake \
    "-DBUILD_COOKBOOK:BOOL=OFF" "-DBUILD_EXAMPLES:BOOL=OFF" "-DBUILD_SHARED_LIBS:BOOL=ON" \
    "-DBUILD_TESTING:BOOL=OFF" "-DOTB_USE_6S:BOOL=OFF" "-DOTB_USE_CURL:BOOL=ON" \
    "-DOTB_USE_GLEW:BOOL=OFF" "-DOTB_USE_GLFW:BOOL=OFF" "-DOTB_USE_GLUT:BOOL=OFF" \
    "-DOTB_USE_GSL:BOOL=OFF" "-DOTB_USE_LIBKML:BOOL=OFF" "-DOTB_USE_LIBSVM:BOOL=ON" \
    "-DOTB_USE_MPI:BOOL=OFF" "-DOTB_USE_MUPARSER:BOOL=ON" "-DOTB_USE_MUPARSERX:BOOL=ON" \
    "-DOTB_USE_OPENCV:BOOL=OFF" "-DOTB_USE_OPENGL:BOOL=OFF" "-DOTB_USE_OPENMP:BOOL=OFF" \
    "-DOTB_USE_QT:BOOL=OFF" "-DOTB_USE_QWT:BOOL=OFF" "-DOTB_USE_SIFTFAST:BOOL=ON" \
    "-DOTB_USE_SPTW:BOOL=OFF" "-DOTB_WRAP_PYTHON:BOOL=ON" "-DCMAKE_BUILD_TYPE=Release" \
    "-DOTB_USE_SHARK:BOOL=OFF" "-DBUILD_EXAMPLES:BOOL=OFF" \
    -DCMAKE_INSTALL_PREFIX="/usr/local/otb" -GNinja .. && \
    ninja install && \
    rm -rf /opt/otb

ENV OTB_APPLICATION_PATH=/usr/local/otb/lib/otb/applications \
    LD_LIBRARY_PATH=/usr/local/otb/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/otb/bin/:$PATH \
    PYTHONPATH=/usr/local/otb/lib/otb/python:$PYTHONPATH \
    GEOTIFF_CSV=/usr/local/otb/share/epsg_csv

COPY gdal-config /usr/local/otb/bin/

# install vlfeat
WORKDIR /opt
RUN git clone https://github.com/vlfeat/vlfeat.git vlfeat

WORKDIR /opt/vlfeat
# https://github.com/vlfeat/vlfeat/issues/214
RUN sed -i 's/default(none)//g' vl/kmeans.c \
    && make \
    && cp bin/glnxa64/libvl.so /usr/local/lib \
    && mkdir -p /usr/local/include/vl \
    && cp -r vl/*.h /usr/local/include/vl
WORKDIR /opt
RUN rm -rf vlfeat
ENV VLFEAT_INCLUDE_DIR=/usr/local/include \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# copy and install cars
WORKDIR /cars
COPY . /cars/
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal
RUN make clean && make install

# source venv/bin/activate in docker mode
ENV VIRTUAL_ENV='/cars/venv'
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Clean pip cache
RUN python -m pip cache purge

# source venv/bin/env_cars.sh
ENV OTB_APPLICATION_PATH=/cars/venv/lib/:$OTB_APPLICATION_PATH \
    PATH=$PATH:/cars/venv/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cars/venv/lib \
    OTB_LOGGER_LEVEL=WARNING \
    OTB_MAX_RAM_HINT=2000 \
    OMP_NUM_THREADS=4 \
    GDAL_CACHEMAX=128 \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
    CARS_NB_WORKERS_PER_PBS_JOB=2

ENV OPJ_NUM_THREADS=$OMP_NUM_THREADS \
    GDAL_NUM_THREADS=$OMP_NUM_THREADS \
    NUMBA_NUM_THREADS=$OMP_NUM_THREADS

# launch cars
ENTRYPOINT ["cars"]
CMD ["-h"]

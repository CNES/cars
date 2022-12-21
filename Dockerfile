# hadolint ignore=DL3007
FROM cnes/cars-deps:latest
LABEL maintainer="CNES"

# copy and install cars with mccnn plugin capabilities installed (but not configured by default)
WORKDIR /cars
COPY . /cars/

RUN make clean && make install-pandora-mccnn

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

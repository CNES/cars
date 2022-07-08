#!/bin/bash
echo " "
echo "===== Demo CARS (with Docker) ====="
echo " "

echo "- Docker must be installed:"
echo "  # docker -v"
docker -v
echo " "

echo "- Get CARS dockerfile image:"
echo "  # docker pull cnes/cars"
docker pull cnes/cars
echo " "

echo "- Get and extract data samples from CARS repository:"
FILE="data_samples.tar.bz2"
URL="https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/${FILE}"
if [ -f ${FILE} ]; then
    echo "  ! File ${FILE} already exists."
else
    # If not present, download data + md5sum.
    echo "  # wget ${URL}"
    echo "  # wget ${URL}.md5sum"
    wget ${URL}
    wget ${URL}.md5sum
fi
# Check md5sum
echo "  # md5sum --status -c ${FILE}.md5sum"
if md5sum --status -c ${FILE}.md5sum; then
    echo "  ! MD5sum check: OK"
else
    # The MD5 sum didn't match
    echo "  ! Md5sum check: KO. Exit."
fi
# Extract cars data samples
echo "  # tar xvfj data_samples.tar.bz2"
tar xvfj data_samples.tar.bz2
echo " "

# CARS Docker Run

echo " Launch CARS with sensor_to_full_resolution_dsm pipeline for img1+img2 and img1+img3 pairs:"
echo "  # docker run -v "$(pwd)"/data_samples:/data cnes/cars /data/configfile.json"
docker run -v "$(pwd)"/data_samples:/data cnes/cars /data/configfile.json
echo " "

# Clean rights on generated data. Otherwise, data cannot be deleted without root access.
docker run -it -v "$(pwd)"/data_samples:/data --entrypoint /bin/bash cnes/cars -c "chown -R '$(id -u):$(id -g)' /data/"

echo "- Show resulting DSM:"
echo "  # ls -l data_samples/outresults/"
ls -l data_samples/outresults/

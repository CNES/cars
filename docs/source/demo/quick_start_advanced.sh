#!/bin/bash
echo " "
echo "===== Demo CARS installed (advanced) ====="
echo " "

echo "- Cars must be installed:"
echo "  # cars -v"
cars -v
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

echo "- Launch CARS with sensor_to_full_resolution_dsm pipeline for img1+img2 and img1+img3 pairs:"
echo "  # cars configfile.json"
cars data_samples/configfile.json
echo " "


echo "- Show resulting DSM:"
echo "  # ls -al data_samples/outresults/"
ls -l data_samples/outresults/

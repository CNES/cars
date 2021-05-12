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

echo "- Launch CARS PREPARE step for img1 and img2 pair:"
echo "  # cars @args_prepare12.txt"
cars @args_prepare12.txt
echo " "

echo "- Launch CARS PREPARE step for img1 and img3 pair:"
echo "  # cars @args_prepare13.txt"
cars @args_prepare13.txt
echo " "

echo "- Launch CARS COMPUTE DSM step:"
echo "  # cars @args_compute.txt"
cars @args_compute.txt
echo " "

echo "- Show resulting DSM:"
echo "  # ls -al data_samples/outcompute/"
ls -l data_samples/outcompute/

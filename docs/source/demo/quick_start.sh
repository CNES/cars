#!/bin/bash
echo " "
echo "===== Demo CARS ====="
echo " "
sleep 2

echo "1. Docker must be installed"
echo "  # docker -v"
docker -v
echo " "
sleep 2

echo "2. Create and go in a CARS test directory :"
echo "  # mkdir test_cars; cd test_cars"
mkdir test_cars; cd test_cars
echo " "
sleep 2


echo "3. Get CARS dockerfile image :"
echo "  # docker pull cnes/cars"
docker pull cnes/cars
echo " "
sleep 2

echo "4. Get and unzip data samples from CARS repository : "
wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/data_samples/data_samples.tar.bz2
tar xvfj data_samples.tar.bz2
rm data_samples.tar.bz2
echo " "
sleep 2

echo "5. Launch CARS PREPARE step for img1 and img2 pair"
echo "  # docker run -v \"\$(pwd)\"/data:/data cnes/cars prepare -i /data/input12.json -o /data/outprepare12 --nb_workers=2"
docker run -v "$(pwd)"/data:/data cnes/cars --loglevel CRITICAL prepare -i /data/input12.json -o /data/outprepare12 --nb_workers=4
echo " "


echo "6. Launch CARS PREPARE step for img1 and img3 pair"
echo "  # docker run -v \"\$(pwd)\"/data:/data cnes/cars prepare -i /data/input13.json -o /data/outprepare13 --nb_workers=2"
docker run -v "$(pwd)"/data:/data cnes/cars --loglevel CRITICAL prepare -i /data/input13.json -o /data/outprepare13 --nb_workers=4
echo " "

echo "7. Launch CARS COMPUTE DSM step"
echo "  # docker run -v \"\$(pwd)\"/data:/data cnes/cars compute_dsm -i /data/outprepare12/content.json /data/outprepare13/content.json -o /data/outcompute/ --nb_workers=2"
docker run -v "$(pwd)"/data:/data cnes/cars --loglevel CRITICAL compute_dsm -i /data/outprepare12/content.json /data/outprepare13/content.json  -o /data/outcompute/ --nb_workers=4
echo " "
# Clean rights on generated data
docker run -it -v "$(pwd)"/data:/data --entrypoint /bin/bash cnes/cars -c "chown -R '$(id -u):$(id -g)' /data/"

echo "8. Show resulting DSM"
echo "  # ls -al data/outcompute/"
ls -al data/outcompute/
sleep 2

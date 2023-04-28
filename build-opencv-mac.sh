#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
OPENCV_DIR=$SCRIPTPATH/opencv

if [ ! -d "$OPENCV_DIR" ]; then
    mkdir -p $OPENCV_DIR
fi
cd $OPENCV_DIR

if [ ! -d "opencv" ]; then
    git clone https://github.com/opencv/opencv.git
    cd opencv && git checkout 4.7.0 && cd ..
fi
if [ ! -d "opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv_contrib && git checkout 4.7.0 && cd ..
fi

mkdir -p build_opencv
cd build_opencv
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_ZLIB=OFF ../opencv
make -j

echo "Installing libraries, this will prompt for sudo"
sudo make install

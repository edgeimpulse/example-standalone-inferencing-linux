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
fi
if [ ! -d "opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
fi

mkdir -p build_opencv
cd build_opencv
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ../opencv
make -j
make install

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
    cd opencv && git checkout 69357b1e88680658a07cffde7678a4d697469f03 && cd .. # v4.5.2
fi
if [ ! -d "opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv_contrib && git checkout f5d7f6712d4ff229ba4f45cf79dfd11c557d56fd && cd ..
fi

mkdir -p build_opencv
cd build_opencv
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ../opencv
make -j3
sudo make install

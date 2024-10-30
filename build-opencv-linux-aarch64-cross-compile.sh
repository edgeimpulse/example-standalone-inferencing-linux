#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=1
            shift # past argument
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

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

CMAKE_ARGS="
    -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
    -DCMAKE_TOOLCHAIN_FILE=../opencv/platforms/linux/aarch64-gnu.toolchain.cmake"

# when `--build-only' flag is passed the install directory will be in
# $OPENCV_DIR/build_opencv/install/
if [[ -z "$BUILD_ONLY" ]]; then
    CMAKE_ARGS+=" -DCMAKE_INSTALL_PREFIX=/usr/local"
fi

mkdir -p build_opencv
cd build_opencv
cmake $CMAKE_ARGS ../opencv
make -j${MAKE_JOBS:-$(nproc)}
make install
ldconfig

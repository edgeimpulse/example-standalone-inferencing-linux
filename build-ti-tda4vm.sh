#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

UNAME=`uname -m`
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

LIB_OUT_ZIP=./output/tidl_model_artifacts.zip
OBJ_OUT_ZIP=./output/tidl_model_artifacts.zip.o

sh ./tflite/linux-ti-tda4vm/download.sh

echo "objcopy zip to object file"
#aarch64-none-linux-gnu-objcopy -I binary -O elf64-littleaarch64 -B aarch64 $LIB_OUT_ZIP $OBJ_OUT_ZIP
aarch64-none-linux-gnu-nm -S -t d $OBJ_OUT_ZIP

echo "clean build"
APP_CUSTOM=1 TARGET_TDA4VM=1 USE_ONNX=1 CC=aarch64-none-linux-gnu-gcc CXX=aarch64-none-linux-gnu-g++ make clean

echo "Build the bin..."
APP_CUSTOM=1 TARGET_TDA4VM=1 USE_ONNX=1 CC=aarch64-none-linux-gnu-gcc CXX=aarch64-none-linux-gnu-g++ make -j

#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH
rm -f libcudart.so libnvinfer.so libnvonnxparser.so tensorrt-shared-libs.zip
wget https://cdn.edgeimpulse.com/build-system/tensorrt-shared-libs.zip
unzip -d . tensorrt-shared-libs.zip
rm tensorrt-shared-libs.zip

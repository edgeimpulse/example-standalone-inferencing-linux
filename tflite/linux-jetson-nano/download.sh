#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

TRT7="tensorrt-shared-libs.zip"
TRT8="shared-libs-jetpack4.6.x.zip"
TRT8_5_2="shared-libs-jetpack5.1.x.zip"
TRT8_6_2="shared-libs-jetpack6.0.0.zip"

download_lib() {
    dir="$1"
    zip_file="$2"

    mkdir -p "$dir"
    cd "$dir"
    rm -rf *.so
    wget https://cdn.edgeimpulse.com/build-system/"$zip_file"
    unzip -q "$zip_file"
    rm -rf "$zip_file"
}

download_lib "$SCRIPTPATH"/trt7 "$TRT7"
download_lib "$SCRIPTPATH"/trt8 "$TRT8"
download_lib "$SCRIPTPATH"/trt8.5.2 "$TRT8_5_2"
download_lib "$SCRIPTPATH"/trt8.6.2 "$TRT8_6_2"

#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

TRT7="tensorrt-shared-libs.zip"
TRT8="shared-libs-jetpack4.6.4.zip"
TRT8_5_2="shared-libs-jetpack5.1.2.zip"

download_lib() {
    dir="$1"
    zip_file="$2"

    cd "$dir"
    rm -rf *.so
    wget https://cdn.edgeimpulse.com/build-system/"$zip_file"
    unzip -q "$zip_file"
    rm -rf "$zip_file"
}

download_lib "$SCRIPTPATH"/trt7 "$TRT7"
download_lib "$SCRIPTPATH"/trt8 "$TRT8"
download_lib "$SCRIPTPATH"/trt8.5.2 "$TRT8_5_2"
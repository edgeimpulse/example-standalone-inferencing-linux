#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/linux-aarch64
if [ ! -f libtensorflowlite_flex_2.19.0.so ]; then
    wget -O libtensorflowlite_flex_2.19.0.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-aarch64/libtensorflowlite_flex_2.19.0.so
fi
if [ ! -f libtensorflowlite_gpu_delegate.so ]; then
    wget -O libtensorflowlite_gpu_delegate.so --show-progress https://cdn.edgeimpulse.com/build-system/gpu-delegates/linux-aarch64/libtensorflowlite_gpu_delegate.so
fi

cd $SCRIPTPATH/linux-armv7
if [ ! -f libtensorflowlite_flex_2.19.0.so ]; then
    wget -O libtensorflowlite_flex_2.19.0.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-armv7/libtensorflowlite_flex_2.19.0.so
fi

cd $SCRIPTPATH/linux-armv7-legacy
if [ ! -f libtensorflowlite_flex_2.16.1.so ]; then
    wget -O libtensorflowlite_flex_2.16.1.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-armv7-legacy/libtensorflowlite_flex_2.16.1.so
fi

cd $SCRIPTPATH/linux-x86
if [ ! -f libtensorflowlite_flex_2.19.0.so ]; then
    wget -O libtensorflowlite_flex_2.19.0.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-x86/libtensorflowlite_flex_2.19.0.so
fi

cd $SCRIPTPATH/mac-arm64
if [ ! -f libtensorflowlite_flex_2.19.0.dylib ]; then
    wget -O libtensorflowlite_flex_2.19.0.dylib --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/mac-arm64/libtensorflowlite_flex_2.19.0.dylib
fi

cd $SCRIPTPATH/mac-x86_64
if [ ! -f libtensorflowlite_flex_2.19.0.dylib ]; then
    wget -O libtensorflowlite_flex_2.19.0.dylib --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/mac-x86_64/libtensorflowlite_flex_2.19.0.dylib
fi

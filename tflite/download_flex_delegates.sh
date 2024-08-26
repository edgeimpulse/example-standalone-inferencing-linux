#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH/mac-x86_64
if [ ! -f libtensorflowlite_flex_2.16.1.dylib ]; then
    wget -O libtensorflowlite_flex_2.16.1.dylib --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/mac-x86_64/libtensorflowlite_flex_2.16.1.dylib
fi

cd $SCRIPTPATH/linux-armv7
if [ ! -f libtensorflowlite_flex_2.16.1.so ]; then
    wget -O libtensorflowlite_flex_2.16.1.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-armv7/libtensorflowlite_flex_2.16.1.so
fi

cd $SCRIPTPATH/linux-aarch64
if [ ! -f libtensorflowlite_flex_2.16.1.so ]; then
    wget -O libtensorflowlite_flex_2.16.1.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-aarch64/libtensorflowlite_flex_2.16.1.so
fi

cd $SCRIPTPATH/linux-x86
if [ ! -f libtensorflowlite_flex_2.16.1.so ]; then
    wget -O libtensorflowlite_flex_2.16.1.so --show-progress https://cdn.edgeimpulse.com/build-system/flex-delegates/linux-x86/libtensorflowlite_flex_2.16.1.so
fi

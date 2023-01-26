#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH
rm -f libvx_tidl_rt.so libti_rpmsg_char.so tidl-shared-libs.zip
wget https://cdn.edgeimpulse.com/build-system/tidl-shared-libs.zip
unzip -d . tidl-shared-libs.zip
rm tidl-shared-libs.zip

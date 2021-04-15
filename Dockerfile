# syntax = docker/dockerfile:experimental
FROM python:3.7.5-stretch

WORKDIR /app

# APT packages
RUN apt update && apt install -y xxd zip clang-3.9

# Install recent CMake
RUN mkdir -p /opt/cmake && \
    cd /opt/cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.sh && \
    sh cmake-3.17.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install a cross-compiler for AARCH64
RUN curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz && \
    mkdir -p /toolchains && \
    tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C /toolchains && \
    rm gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz

ENV PATH="/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/:${PATH}"

# Linux runner (Linux AARCH64)
COPY ./ /linux-impulse-runner/linux_aarch64
RUN cd /linux-impulse-runner/linux_aarch64 && \
    CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ APP_CUSTOM=1 TARGET_JETSON_NANO=1 make clean
# RUN cd /linux-impulse-runner/linux_aarch64 && \
#     CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ APP_CUSTOM=1 TARGET_JETSON_NANO=1 make -j

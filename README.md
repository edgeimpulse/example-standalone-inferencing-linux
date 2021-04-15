# Edge Impulse Linux SDK for C++

This library lets you run machine learning models and collect sensor data on Linux machines using C++. This SDK is part of [Edge Impulse](https://www.edgeimpulse.com) where we enable developers to create the next generation of intelligent device solutions with embedded machine learning. [Start here to learn more and train your first model](https://docs.edgeimpulse.com).

## Installation guide

1. Install GNU Make and a recent C++ compiler (tested with GCC 8 on the Raspberry Pi, and Clang on other targets).
1. Clone this repository and initialize the submodules:

    ```
    $ git clone https://github.com/edgeimpulse/example-standalone-inferencing-linux
    $ cd example-standalone-inferencing-linux && git submodule update --init --recursive
    ```

1. If you want to use the audio or camera examples, you'll need to install libasound2 and OpenCV 4. You can do so via:

    **Linux**

    ```
    $ sudo apt install libasound2
    $ sh build-opencv-linux.sh          # only needed if you want to run the camera example
    ```

    **macOS**

    ```
    $ sh build-opencv-mac.sh            # only needed if you want to run the camera example
    ```

    Note that you cannot run any of the audio examples on macOS, as these depend on libasound2, which is not available there.

## Collecting data

Before you can classify data you'll first need to collect it. If you want to collect data from the camera or microphone on your system you can use the Edge Impulse CLI, and if you want to collect data from different sensors (like accelerometers or proprietary control systems) you can do so in a few lines of code.

### Collecting data from the camera or microphone

To collect data from the camera or microphone, follow the [getting started guide](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux) for your development board.

### Collecting data from other sensors

To collect data from other sensors you'll need to write some code to collect the data from an external sensor, wrap it in the Edge Impulse Data Acquisition format, and upload the data to the Ingestion service. [Here's an end-to-end example](https://github.com/edgeimpulse/example-standalone-inferencing-linux/blob/master/source/collect.cpp) that you can build via:

```
$ APP_COLLECT=1 make -j
```

## Classifying data

This repository comes with three classification examples:

* [custom](https://github.com/edgeimpulse/example-standalone-inferencing-linux/blob/master/source/custom.cpp) - classify custom sensor data (`APP_CUSTOM=1`).
* [audio](https://github.com/edgeimpulse/example-standalone-inferencing-linux/blob/master/source/audio.cpp) - realtime audio classification (`APP_AUDIO=1`).
* [camera](https://github.com/edgeimpulse/example-standalone-inferencing-linux/blob/master/source/camera.cpp) - realtime image classification (`APP_CAMERA=1`).

To build an application:

1. [Train an impulse](https://docs.edgeimpulse.com/docs).
1. Export your trained impulse as a C++ Library from the Edge Impulse Studio (see the **Deployment** page) and copy the folders into this repository.
1. Build the application via:

    ```
    $ APP_CUSTOM=1 make -j
    ```

    Replace `APP_CUSTOM=1` with the application you want to build. See 'Hardware acceleration' below for the hardware specific flags. You probably want these.

1. The application is in the build directory:

    ```
    $ ./build/custom
    ```

### Hardware acceleration

For many targets there is hardware acceleration available. To enable this:

**Raspberry Pi 4 (and other Armv7l Linux targets)**

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1 make -j
```

**Jetson Nano (and other AARCH64 targets)**

See the [TensoRT](#tensorrt) section below for information on enabling GPUs. To build with hardware extensions for running on the CPU:

1. Install Clang:

    ```
    $ sudo apt install -y clang
    ```

1. Build with the following flags:

    ```
    $ APP_CUSTOM=1 TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 CC=clang CXX=clang++ make -j
    ```

**Intel-based Macs**

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 make -j
```

### TensorRT

On the Jetson Nano you can also build with support for TensorRT, this fully leverages the GPU on the Jetson Nano. Unfortunately this is currently not available for object detection models - which is why this is not enabled by default. To build with TensorRT:

1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the 'TensorRT library', and the 'float32' optimizations.
1. Build the library and copy the folders into this repository.
1. Download the shared libraries via:

    ```
    $ sh ./tflite/linux-jetson-nano/download.sh
    ```

1. Build your application with:

    ```
    $ APP_CUSTOM=1 TARGET_JETSON_NANO=1 make -j
    ```

Note that there is significant ramp up time required for TensorRT. The first time you run a new model the model needs to be optimized - which might take up to 30 seconds, then on every startup the model needs to be loaded in - which might take up to 5 seconds. After this, the GPU seems to be warming up, so expect full performance about 2 minutes in. To do a fair performance comparison you probably want to use the custom application (no camera / microphone overhead) and run the classification in a loop.

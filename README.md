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
    $ sudo apt install libasound2-dev
    $ sh build-opencv-linux.sh          # only needed if you want to run the camera example
    ```

    > Note: If you can't find `alsa/asoundlib.h` during building you may need to reboot after installing libasound2 to see effects.

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

**Armv7l Linux targets**

> e.g. Raspberry Pi 4

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1 make -j
```

**AARCH64 Linux targets**

> e.g. NVIDIA Jetson Nano, Renesas RZ/V2L

> See the [AARCH64 with AI Acceleration](#aarch64-with-ai-acceleration) section below for information on enabling hardware (AI) acceleration for your AARCH64 Linux target.

1. Install Clang:

    ```
    $ sudo apt install -y clang
    ```

1. Build with the following flags:

    ```
    $ APP_CUSTOM=1 TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1 CC=clang CXX=clang++ make -j
    ```

**x86 Linux targets**

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_LINUX_X86=1 USE_FULL_TFLITE=1 make -j
```

**Intel-based Macs**

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 make -j
```

**M1-based Macs**

Build with the following flags:

```
$ APP_CUSTOM=1 TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 arch -x86_64 /usr/bin/make -j
```

Note that this does build an x86 binary, but it runs very fast through Rosetta.

### AARCH64 with AI Acceleration

#### NVIDIA Jetson Nano - TensorRT

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

#### Renesas RZV2L - DRP-AI

On the Renesas RZ/V2L you can also build with support for DRP-AI, this fully leverages the DRP and AI-MAC on the Renesas RZ/V2L.

1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the 'DRP-AI library', and the 'float32' optimizations.

> Note: currently only RGB MobileNetV2 Image Classification, FOMO and [YOLOv5 (v5)](https://github.com/edgeimpulse/yolov5/tree/v5) models supported.

1. Build the library and copy the folders into this repository.

1. Build your application with:

    ```
    $ TARGET_RENESAS_RZV2L=1 make -j
    ```

#### BrainChip AKD1000

You can build EIM or other inferencing examples with the support for BrainChip AKD1000 NSoC. Currently, it is supported on Linux boards with x86_64 or AARCH64 architectures.
To build the application with support for AKD1000 NSoC, you need a Python development library on your build system.

1. Install dependencies
    Check if you have an output for `python3-config --cflags` command. If you get `bash: command not found: python3-config`, then try to install it with
    ```
    $ apt install -y python3-dev`
    ```
    Also, install the Python `akida` library
    ```
    $ pip3 install akida
    ```
1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the `Meta TF Model` and build.
1. Extract the content of the downloaded zip archive into this directory.
1. Build your application with `USE_AKIDA=1`, for example:

    ```
    $ USE_AKIDA=1 APP_EIM=1 TARGET_LINUX_AARCH64=1 make -j
    ```

In case of any issues during runtime, check [Troubleshooting](https://docs.edgeimpulse.com/docs/development-platforms/officially-supported-ai-accelerators/akd1000#troubleshooting) section in our official documentation for AKD1000 NSoc.

#### Texas Instruments TDA4VM (AM68PA), AM62A, AM68A - TI Deep Learning (TIDL)

You can also build with support for TIDL, this fully leverages the Deep Learning Accelerator on the Texas Instruments TDA4VM (AM68PA), AM62A, AM68A.

##### TDA4VM (AM68PA)

1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the 'TIDL-RT Library', and the 'float32' optimizations.
1. Build the library and copy the folders into this repository.
1. Build your (.eim) application:

    ```
    $ APP_EIM=1 TARGET_TDA4VM=1 make -j
    ```

To build for ONNX runtime:

```
$ APP_EIM=1 TARGET_TDA4VM=1 USE_ONNX=1 make -j
```

##### TI AM62A

1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the 'TIDL-RT Library (AM62A)', and the 'float32' optimizations.
1. Build the library and copy the folders into this repository.
1. Build your (.eim) application:

    ```
    $ APP_EIM=1 TARGET_AM62A=1 make -j
    ```

To build for ONNX runtime:

```
$ APP_EIM=1 TARGET_AM62A=1 USE_ONNX=1 make -j
```

##### TI AM68A

1. Go to the **Deployment** page in the Edge Impulse Studio.
1. Select the 'TIDL-RT Library (AM68A)', and the 'float32' optimizations.
1. Build the library and copy the folders into this repository.
1. Build your (.eim) application:

    ```
    $ APP_EIM=1 TARGET_AM68A=1 make -j
    ```

To build for ONNX runtime:

```
$ APP_CUSTOM=1 TARGET_AM68A=1 USE_ONNX=1 make -j
```

## Building .eim files

To build Edge Impulse for Linux models ([eim files](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux#eim-models)) that can be used by the Python, Node.js or Go SDKs build with `APP_EIM=1`:

```
$ APP_EIM=1 make -j
```

The model will be placed in `build/model.eim` and can be used directly by your application.

## Troubleshooting

### Failed to allocate TFLite arena (0 bytes)

If you see the error above, then you should be building with [hardware acceleration enabled](#hardware-acceleration). The reason is that when running without hardware optimizations enabled we run under TensorFlow Lite Micro and your model is not supported there (most likely you have unsupported ops or your model is too big for TFLM) - which is why we couldn't determine the arena size. Enabling hardware acceleration switches to full TensorFlow Lite.

### Make sure you apply/link the Flex delegate before inference.

On Linux platforms without a GPU or neural accelerator your model is ran using TensorFlow Lite. Not every model can be represented using native TensorFlow Lite operators; and for these models 'Flex' ops are injected in the model. To run these models you'll need to link with the flex delegate shared library when compiling your model, and then have this library installed on any device where you run the model. If this is the case you'll seen an error like:

```
ERROR: Regular TensorFlow ops are not supported by this interpreter. Make sure you apply/link the Flex delegate before inference.
ERROR: Node number 33 (FlexErf) failed to prepare.
```

To solve this:

1. Download the flex delegates shared library via:

    ```
    bash tflite/download_flex_delegates.sh
    ```

2. Build your application with `LINK_TFLITE_FLEX_LIBRARY=1` .
3. Copy the shared library for your platform to `/usr/lib` or `/usr/local/lib` to run your application (see [Docs: Flex delegates](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux/flex-delegates)).

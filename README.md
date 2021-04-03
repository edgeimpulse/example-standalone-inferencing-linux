# Edge Impulse Linux SDK for C++

This library lets you run machine learning models and collect sensor data on Linux machines using C++. This SDK is part of [Edge Impulse](https://www.edgeimpulse.com) where we enable developers to create the next generation of intelligent device solutions with embedded machine learning. [Start here to learn more and train your first model](https://docs.edgeimpulse.com).

This repository comes with four examples:

* [custom](source/custom.cpp) - classify custom sensor data (`APP_CUSTOM=1`).
* [audio](source/audio.cpp) - realtime audio classification (`APP_AUDIO=1`).
* [camera](source/camera.cpp) - realtime image classification (`APP_CAMERA=1`).
* [collect](source/collect.cpp) - collect custom sensor data (`APP_COLLECT=1`).

To build an application:

1. [Train an impulse](https://docs.edgeimpulse.com/docs).
1. Export your trained impulse as a C++ Library from the Edge Impulse Studio (see the **Deployment** page) and copy them into this repository.
1. Install the dependencies:

    **Linux**

    ```
    $ sudo apt install libasound2 libopencv-dev
    ```

    **macOS**

    ```
    $ sh build-opencv-mac.sh
    ```

    Note that you cannot run any of the audio examples on macOS, as these depend on libasound2, which is not available there.

1. Build the application via:

    ```
    $ APP_CUSTOM=1 make -j
    ```

    Replace `APP_CUSTOM=1` with the application you want to build.

1. The application is in the build directory:

    ```
    $ ./build/custom
    ```

### Hardware acceleration

For many targets there is hardware acceleration available. To enable this:

* Raspberry Pi 4 and other Armv7 Linux targets: Build with `TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1` flags.
* AARCH64 Linux targets: Build with `TARGET_LINUX_AARCH64=1 USE_FULL_TFLITE=1` flags.
* Intel-based Macs: Build with `TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1` flags.

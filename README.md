# Edge Impulse Example: stand-alone inferencing (C++)

This builds and runs an exported impulse locally on your machine. See the documentation at [Running your impulse locally](https://docs.edgeimpulse.com/docs/running-your-impulse-locally). There is also a [C version](https://github.com/edgeimpulse/example-standalone-inferencing-c) of this application.

## How to run (Rpi4)

1. Copy everything over to a Raspberry Pi 4.
1. Compile:

    ```
    $ TARGET_LINUX_ARMV7=1 USE_FULL_TFLITE=1 make -j
    ```

1. Run:

    ```
    $ ./build/edge-impulse-standalone features.txt
    ```

## How to run (macOS x86)

1. Compile:

    ```
    $ TARGET_MAC_X86_64=1 USE_FULL_TFLITE=1 make -j
    ```

1. Run:

    ```
    $ ./build/edge-impulse-standalone features.txt
    ```

The `trt8.5.2/libei_debug.a` file was built from: https://github.com/edgeimpulse/TensorRT/tree/libeirt-tensorrt-8.5.2 - build using:

```
$ sh docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt --cuda 11.8.0
```

The `trt8/libei_debug.a` file was built from: https://github.com/edgeimpulse/TensorRT/tree/libeirt - build using:

```
$ sh docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt --cuda 10.2
```

You can find the library in `/workspace/TensorRT/build/out/libei_debug.a` in the container. It is also automatically copied after a successful build to the /docker folder.

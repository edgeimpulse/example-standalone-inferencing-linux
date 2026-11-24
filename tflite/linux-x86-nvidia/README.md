The `libei_debug.a` file was built from: https://github.com/edgeimpulse/TensorRT/tree/libeirt-tensorrt-10.16.1 - build using:

```
$ sh docker/build.sh --file docker/ubuntu-22.04.Dockerfile --tag tensorrt --cuda 13.2
```

You can find the library in `/workspace/TensorRT/build/out/libei_debug.a` in the container. It is also automatically copied after a successful build to the /docker folder.

# Segfault on TFLite C++ w/ QNN

This ran on IQ-9075 EVK with setup instructions from https://qc-ai-test.gitbook.io/qc-ai-test-docs/device-setup/iq9075-evk. So set up using Ubuntu 24, with QAIRT SDK from Qualcomm PPA.

```bash
$ cat /proc/version
# Linux version 6.8.0-1054-qcom (buildd@bos03-arm64-053) (aarch64-linux-gnu-gcc-13 (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0, GNU ld (GNU Binutils for Ubuntu) 2.42) #54-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep  8 15:48:39 UTC 2025

$ apt-cache policy libqnn1
# libqnn1:
#   Installed: 2.38.0.250901-0ubuntu1
#   Candidate: 2.38.0.250901-0ubuntu1
#   Version table:
#  *** 2.38.0.250901-0ubuntu1 500
#         500 https://ppa.launchpadcontent.net/ubuntu-qcom-iot/qcom-ppa/ubuntu noble/main arm64 Packages
#         100 /var/lib/dpkg/status

# Then downloaded extra headers etc not in apt via:
$ wget -qO- https://cdn.edgeimpulse.com/qc-ai-docs/device-setup/install_ai_runtime_sdk.sh | bash

# Set QAIRT root folder
export QNN_SDK_ROOT="/home/ubuntu/qairt/2.38.0.250901"


# THIS IS A WORKING QNN SETUP! Python LiteRT w/ QNN works (just like other QNN stuff)
```

Build this application:

```bash
rm -f source/*.o
APP_CUSTOM=1 TARGET_LINUX_AARCH64=1 USE_QUALCOMM_QNN=1 make -j`nproc`
./build/custom features.txt

# ====== DDR bandwidth summary ======
# spill_bytes=0
# fill_bytes=0
# write_total_bytes=65536
# read_total_bytes=8542208
#
#  <W> Logs will be sent to the system's default channel
# Predictions (DSP: 1 ms., Classification: 0 ms., Anomaly: 0 ms.):
# #Object detection results:
# jan (0.893009) [ x: 41, y: 11, width: 193, height: 254 ]
# thumbsup (0.705765) [ x: 190, y: 91, width: 116, height: 215 ]
# Segmentation fault (core dumped) / Bus error (core dumped)
```

Run with lldb:

```
lldb -o run -- ./build/custom features.txt

# Process 3887 launched: '/home/ubuntu/example-standalone-inferencing-linux/build/custom' (aarch64)
# Process 3887 stopped
# * thread #1, name = 'custom', stop reason = signal SIGSEGV: address not mapped to object (fault address: 0xaaaab0245)
#     frame #0: 0x0000ffffe5169250 libQnnHtpPrepare.so`___lldb_unnamed_symbol72386 + 672
# libQnnHtpPrepare.so`___lldb_unnamed_symbol72386:
# ->  0xffffe5169250 <+672>: ldr    x1, [x1, #0x8]

bt

# * thread #1, name = 'custom', stop reason = signal SIGSEGV: address not mapped to object (fault address: 0xaaaab0245)
#   * frame #0: 0x0000ffffe5169250 libQnnHtpPrepare.so`___lldb_unnamed_symbol72386 + 672
#     frame #1: 0x0000ffffe516a0a4 libQnnHtpPrepare.so`std::_Rb_tree<unsigned int, std::pair<unsigned int const, hnnx::DataWriter>, std::_Select1st<std::pair<unsigned int const, hnnx::DataWriter>>, std::less<unsigned int>, std::allocator<std::pair<unsigned int const, hnnx::DataWriter>>>::erase(unsigned int const&) + 100
#     frame #2: 0x0000ffffe516a258 libQnnHtpPrepare.so`___lldb_unnamed_symbol72396 + 88
#     frame #3: 0x0000ffffe51f9c24 libQnnHtpPrepare.so`GraphPrepare::~GraphPrepare() + 2212
#     frame #4: 0x0000ffffe51f9c50 libQnnHtpPrepare.so`GraphPrepare::~GraphPrepare() + 16
#     frame #5: 0x0000ffffe42fbd44 libQnnHtpPrepare.so`___lldb_unnamed_symbol14446 + 52
#     frame #6: 0x0000fffff79af228 libc.so.6`__run_exit_handlers(status=1, listp=0x0000fffff7b20670, run_list_atexit=true, run_dtors=true) at exit.c:108:8
#     frame #7: 0x0000fffff79af30c libc.so.6`__GI_exit(status=<unavailable>) at exit.c:138:3
#     frame #8: 0x0000fffff79984c8 libc.so.6`__libc_start_call_main(main=(custom`main at custom.cpp:107:33), argc=2, argv=0x0000ffffffffefc8) at libc_start_call_main.h:74:3
#     frame #9: 0x0000fffff7998598 libc.so.6`__libc_start_main_impl(main=(custom`main at custom.cpp:107:33), argc=2, argv=0x0000ffffffffefc8, init=<unavailable>, fini=<unavailable>, rtld_fini=<unavailable>,
```

Can run same binary w/ XNNPACK rather than QNN:

```bash
APP_CUSTOM=1 TARGET_LINUX_AARCH64=1 make clean
APP_CUSTOM=1 TARGET_LINUX_AARCH64=1 make -j`nproc`
./build/custom features.txt

# Will run and not have any segfault / busfault
```

This is where inference is done, and the QNN delegates are loaded:

```
edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h
```

## Python?

Works fine. See [python/](python/) - uses exactly same model.

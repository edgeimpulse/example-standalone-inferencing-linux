#######################################
# USE_MEMRYX
#######################################

ifeq (${USE_MEMRYX},1)
CFLAGS += -Iedge-impulse-sdk/third_party/gemmlowp
LDFLAGS += -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread -lrt
ifeq (${TARGET_LINUX_AARCH64},1)
CFLAGS += -DDISABLEFLOAT16
LDFLAGS += -L./tflite/linux-aarch64
LDFLAGS += -L /usr/lib/aarch64-linux-gnu/ -lmemx
else ifeq (${TARGET_LINUX_X86},1)
ifdef (${EI_CLASSIFIER_USE_MEMRYX_SOFTWARE},1)
CFLAGS += $(shell python3-config --cflags)
CFLAGS += -DPYBIND11_DETAILED_ERROR_MESSAGES
LDFLAGS += -rdynamic $(shell python3-config --ldflags --embed)
else
LDFLAGS += -L./tflite/linux-x86
LDFLAGS += -lmemx
endif # USE_MEMRYX_SOFTWARE
endif # USE_MEMRYX && TARGET_LINUX_X86
endif # USE_MEMRYX

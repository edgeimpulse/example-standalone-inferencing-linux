#######################################
# USE_FULL_TFLITE
#######################################

ifeq (${USE_FULL_TFLITE},1)
CFLAGS += -DEI_CLASSIFIER_USE_FULL_TFLITE=1
CFLAGS += -Iedge-impulse-sdk/tensorflow-lite
CCSOURCES += $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/custom/*.cc)

ifeq (${TFLITE_VERSION},2.19.0)
TFLITE_LIBS_FLAGS = -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lmicrokernels-prod -lcpuinfo -lpthreadpool -lpthread
else ifeq (${TFLITE_VERSION},2.16.1)
TFLITE_LIBS_FLAGS = -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
else ifeq (${TFLITE_VERSION},2.7.0)
TFLITE_LIBS_FLAGS = -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
else
$(error Unsupported TFLITE_VERSION ${TFLITE_VERSION})
endif # TFLITE_VERSION

ifeq (${TARGET_LINUX_ARMV7},1)
LDFLAGS += -L./tflite/linux-armv7 -Wl,--no-as-needed -ldl -lrt $(TFLITE_LIBS_FLAGS)
endif # TARGET_LINUX_ARMV7

ifeq (${TARGET_LINUX_ARMV7_LEGACY},1)
TFLITE_VERSION = 2.16.1
TFLITE_LIBS_FLAGS = -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lflatbuffers -lruy -lXNNPACK -lpthreadpool -lpthread -lcpuinfo
LDFLAGS += -L./tflite/linux-armv7-legacy -Wl,--no-as-needed -ldl $(TFLITE_LIBS_FLAGS) -lrt
endif # TARGET_LINUX_ARMV7_LEGACY

# TARGET_LINUX_AARCH64 || TARGET_TDA4VM
ifneq ($(filter 1,${TARGET_LINUX_AARCH64} ${TARGET_TDA4VM}),)
CFLAGS += -DDISABLEFLOAT16
ifeq (${USE_GPU_DELEGATES},1)
CFLAGS += -DEI_CLASSIFIER_USE_GPU_DELEGATES=1
LDFLAGS += -L./tflite/linux-aarch64 -Wl,--no-as-needed -ltensorflowlite_gpu_delegate -ldl -lrt $(TFLITE_LIBS_FLAGS) -lkleidiai
# else TFLITE_VERSION == 2.19.0
else ifeq (${TFLITE_VERSION},2.19.0)
LDFLAGS += -L./tflite/linux-aarch64 -Wl,--no-as-needed -ldl -lrt $(TFLITE_LIBS_FLAGS) -lkleidiai
else
LDFLAGS += -L./tflite/linux-aarch64 -Wl,--no-as-needed -ldl -lrt $(TFLITE_LIBS_FLAGS)
endif # USE_GPU_DELEGATES
endif # TARGET_LINUX_AARCH64 || TARGET_TDA4VM

ifeq (${TARGET_TDA4VM},1)
CFLAGS += -DDISABLEFLOAT16
LDFLAGS += -L./tflite/linux-aarch64 -Wl,--no-as-needed -ldl -lrt $(TFLITE_LIBS_FLAGS) -lclog
endif # TARGET_TDA4VM
ifeq (${TARGET_LINUX_X86},1)
LDFLAGS += -L./tflite/linux-x86 -Wl,--no-as-needed -ldl -lrt $(TFLITE_LIBS_FLAGS)
endif # TARGET_LINUX_X86
ifeq (${TARGET_MAC_X86_64},1)
LDFLAGS += -L./tflite/mac-x86_64 $(TFLITE_LIBS_FLAGS)
endif # TARGET_MAC_X86_64
ifeq (${TARGET_MAC_ARM64},1)
CFLAGS += -target arm64-apple-darwin
LDFLAGS += -L./tflite/mac-arm64 $(TFLITE_LIBS_FLAGS) -lkleidiai
endif # TARGET_MAC_ARM64
endif # USE_FULL_TFLITE

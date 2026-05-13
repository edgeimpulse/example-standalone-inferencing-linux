#######################################
# USE_AKIDA
#######################################

ifeq (${USE_AKIDA},1)
CFLAGS += -DEI_CLASSIFIER_USE_FULL_TFLITE=1
CFLAGS += -DPYBIND11_DETAILED_ERROR_MESSAGES # add more detailed pybind error descriptions
CFLAGS += -Iedge-impulse-sdk/tensorflow-lite
CFLAGS += -Iedge-impulse-sdk/third_party/gemmlowp
LDFLAGS += -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread -lrt
ifeq (${TARGET_LINUX_AARCH64},1)
CFLAGS += $(shell $(PYTHON_CROSS_PATH)python3-config --cflags)
LDFLAGS += -L./tflite/linux-aarch64
LDFLAGS += $(shell $(PYTHON_CROSS_PATH)python3-config --ldflags --embed)
else ifeq (${TARGET_LINUX_X86},1)
CFLAGS += $(shell python3-config --cflags)
LDFLAGS += -L./tflite/linux-x86
LDFLAGS += $(shell python3-config --ldflags --embed)
endif # TARGET_LINUX_AARCH64 || TARGET_LINUX_X86
endif # USE AKIDA

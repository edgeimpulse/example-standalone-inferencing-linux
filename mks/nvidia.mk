#######################################
# JETSON targets
#######################################

ifeq (${TARGET_JETSON_ORIN},1)
TARGET_JETSON_COMMON=1
TENSORRT_VERSION?=8.5.2
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
endif

ifeq (${TARGET_JETSON_NANO},1)
TARGET_JETSON=1
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
endif

ifeq (${TARGET_JETSON},1)
TARGET_JETSON_COMMON=1
TENSORRT_VERSION?=8
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
endif

ifeq (${TARGET_JETSON_COMMON},1)
TENSORRT_VERSION ?=8
$(info TENSORRT_VERSION is ${TENSORRT_VERSION})
ifeq (${TENSORRT_VERSION},8.6.2)
TRT_LDFLAGS += -lei_debug -Ltflite/linux-jetson-nano/trt8.6.2/
else ifeq (${TENSORRT_VERSION},8.5.2)
TRT_LDFLAGS += -lei_debug -Ltflite/linux-jetson-nano/trt8.5.2/
else ifeq (${TENSORRT_VERSION},8)
TRT_LDFLAGS += -lei_debug -Ltflite/linux-jetson-nano/trt8/
else
$(error Invalid TensorRT version)
endif # TENSORRT_VERSION
TRT_LDFLAGS += -lcudart -lnvinfer -lnvonnxparser
LDFLAGS += $(TRT_LDFLAGS) -lstdc++fs -Ltflite/linux-jetson-nano/ -Wl,--warn-unresolved-symbols,--unresolved-symbols=ignore-in-shared-libs
endif # TARGET_JETSON_COMMON

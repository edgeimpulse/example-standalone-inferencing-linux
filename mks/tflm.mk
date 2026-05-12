#######################################
# Check if USE_FULL_TFLITE USE_AKIDA USE_ONNX are not set
#######################################
USE_FULL_TFLITE?=0
USE_AKIDA?=0
USE_ONNX?=0

ifeq (${USE_FULL_TFLITE},0)
ifeq (${USE_AKIDA},0)
ifeq (${USE_ONNX},0)
CFLAGS += -DTF_LITE_DISABLE_X86_NEON=1
CSOURCES += edge-impulse-sdk/tensorflow/lite/c/common.c
CCSOURCES += $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/internal/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/memory_planner/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/core/api/*.cc)
endif
endif
endif

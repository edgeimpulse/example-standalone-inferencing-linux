#######################################
# TI targets
#######################################

ifeq (${TARGET_AM68PA},1)
TARGET_TDA4VM=1
endif

ifeq (${TARGET_AM62A},1)
TARGET_TDA4VM=1
endif

ifeq (${TARGET_AM68A},1)
TARGET_TDA4VM=1
endif

ifeq (${TARGET_TDA4VM},1)
CFLAGS += -I${TIDL_TOOLS_PATH} -I${TIDL_TOOLS_PATH}/osrt_deps
LDFLAGS += -L./tidl-rt/linux-aarch64 -lvx_tidl_rt -lti_rpmsg_char -lrt
TFLITE_VERSION=2.7.0
ifeq (${USE_ONNX},1)
ONNX_SUB = onnx_1.7.0_x86_u18
CFLAGS += -I${TIDL_TOOLS_PATH}/osrt_deps/${ONNX_SUB}/onnxruntime/include -I${TIDL_TOOLS_PATH}/osrt_deps/${ONNX_SUB}/onnxruntime/include/onnxruntime -I${TIDL_TOOLS_PATH}/osrt_deps/${ONNX_SUB}/onnxruntime/include/onnxruntime/core/session
CFLAGS += -DDISABLEFLOAT16 -DXNN_ENABLE=0
LDFLAGS += -L${TIDL_TOOLS_PATH} -Wl,--no-as-needed -lonnxruntime -ldl -ldlr -lpthread -lrt
endif
endif # TARGET_TDA4VM

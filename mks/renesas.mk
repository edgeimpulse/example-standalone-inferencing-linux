#######################################
# RENESAS targets
#######################################

ifeq (${TARGET_RENESAS_RZV2H},1)
TARGET_RENESAS_RZV2L=1
endif

ifeq (${USE_TVM},1)

ifndef TVM_HOME
$(error TVM_HOME variable not set)
endif

CFLAGS += -I${TVM_HOME}/include
CFLAGS += -I${TVM_HOME}/3rdparty/dlpack/include
CFLAGS += -I${TVM_HOME}/3rdparty/dmlc-core/include
CFLAGS += -I${TVM_HOME}/3rdparty/compiler-rt
LDFLAGS += -L${TVM_HOME}/build_runtime/ -ltvm_runtime
endif

ifeq (${TARGET_RENESAS_RZV2L},1)
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
TFLITE_VERSION=2.16.1
endif

ifeq (${TARGET_RENESAS_RZG2L},1)
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
TFLITE_VERSION=2.16.1
endif


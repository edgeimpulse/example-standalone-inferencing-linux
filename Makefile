EI_SDK?=edge-impulse-sdk
PYTHON_CROSS_PATH?=

UNAME_S := $(shell uname -s)

CFLAGS +=  -Wall -g -Wno-strict-aliasing
CFLAGS += -I.
CFLAGS += -Isource
CFLAGS += -Imodel-parameters
CFLAGS += -Itflite-model
CFLAGS += -Ithird_party/
CFLAGS += -Iutils/
CFLAGS += -Os
CFLAGS += -DNDEBUG
CFLAGS += -DEI_CLASSIFIER_ENABLE_DETECTION_POSTPROCESS_OP=1
CFLAGS += -g
ifeq (${CC}, clang)
	CFLAGS += -Wno-asm-operand-widths
endif
CXXFLAGS += -std=c++17
LDFLAGS += -lm -lstdc++

CSOURCES = $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/TransformFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/CommonTables/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/BasicMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/ComplexMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/FastMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/SupportFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/MatrixFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/*.c)
CXXSOURCES = $(wildcard tflite-model/*.cpp) $(wildcard edge-impulse-sdk/dsp/kissfft/*.cpp) $(wildcard edge-impulse-sdk/dsp/dct/*.cpp) $(wildcard ./edge-impulse-sdk/dsp/memory.cpp) $(wildcard edge-impulse-sdk/porting/posix/*.c*) $(wildcard edge-impulse-sdk/porting/mingw32/*.c*)
CCSOURCES =

ifeq (${TARGET_RENESAS_RZV2L},1)
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
endif

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
LDFLAGS +=  -L./tidl-rt/linux-aarch64 -lti_rpmsg_char -lvx_tidl_rt

ifeq (${USE_ONNX},1)
CFLAGS += -I${TIDL_TOOLS_PATH}/osrt_deps/onnxruntime/include -I${TIDL_TOOLS_PATH}/osrt_deps/onnxruntime/include/onnxruntime -I${TIDL_TOOLS_PATH}/osrt_deps/onnxruntime/include/onnxruntime/core/session
CFLAGS += -DDISABLEFLOAT16 -DXNN_ENABLE=0
LDFLAGS += -Wl,--no-as-needed -lonnxruntime -ldl -ldlr -lpthread #-lpcre -lffi -lz -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -ltbb -ljpeg -lwebp -lpng16 -ltiff -lyaml-cpp

else
USE_FULL_TFLITE=1
TARGET_LINUX_AARCH64=1
endif
endif

ifeq (${TARGET_JETSON_ORIN},1)
TARGET_JETSON_COMMON=1
TENSORRT_VERSION=8.5.2
endif

ifeq (${TARGET_JETSON_NANO},1)
TARGET_JETSON=1
endif

ifeq (${TARGET_JETSON},1)
TARGET_JETSON_COMMON=1
TENSORRT_VERSION?=8
endif

ifeq (${USE_FULL_TFLITE},1)
CFLAGS += -DEI_CLASSIFIER_USE_FULL_TFLITE=1
CFLAGS += -Itensorflow-lite/
CCSOURCES += $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/custom/*.cc)

ifeq (${TARGET_LINUX_ARMV7},1)
LDFLAGS += -L./tflite/linux-armv7 -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lflatbuffers -lruy -lXNNPACK -lpthreadpool -lpthread -lcpuinfo
endif # TARGET_LINUX_ARMV7
ifeq (${TARGET_LINUX_AARCH64},1)
CFLAGS += -DDISABLEFLOAT16
LDFLAGS += -L./tflite/linux-aarch64 -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
endif # TARGET_LINUX_AARCH64
ifeq (${TARGET_LINUX_X86},1)
LDFLAGS += -L./tflite/linux-x86 -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
endif # TARGET_LINUX_X86
ifeq (${TARGET_MAC_X86_64},1)
LDFLAGS += -L./tflite/mac-x86_64 -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthreadpool
endif # TARGET_MAC_X86_64
ifeq (${TARGET_MAC_ARM64},1)
LDFLAGS += -L./tflite/mac-arm64 -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthreadpool
endif # TARGET_MAC_ARM64

ifeq (${LINK_TFLITE_FLEX_LIBRARY},1)
LDFLAGS += -ltensorflowlite_flex_2.16.1
endif

else ifeq (${USE_AKIDA},1) # USE_FULL_TFLITE
CFLAGS += -DEI_CLASSIFIER_USE_FULL_TFLITE=1
CFLAGS += -DPYBIND11_DETAILED_ERROR_MESSAGES # add more detailed pybind error descriptions
CFLAGS += -Itensorflow-lite
CFLAGS += -Iedge-impulse-sdk/third_party/gemmlowp
LDFLAGS += -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
ifeq (${TARGET_LINUX_AARCH64},1)
CFLAGS += $(shell $(PYTHON_CROSS_PATH)python3-config --cflags)
LDFLAGS += -L./tflite/linux-aarch64
LDFLAGS += $(shell $(PYTHON_CROSS_PATH)python3-config --ldflags --embed)
else ifeq (${TARGET_LINUX_X86},1) # TARGET_LINUX_AARCH64
CFLAGS += $(shell python3-config --cflags)
LDFLAGS += -L./tflite/linux-x86
LDFLAGS += $(shell python3-config --ldflags --embed)
endif # TARGET_LINUX_X86

else # not USE_FULL_TFLITE and not USE_AKIDA

CFLAGS += -DTF_LITE_DISABLE_X86_NEON=1
CSOURCES += edge-impulse-sdk/tensorflow/lite/c/common.c
CCSOURCES += $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/internal/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/memory_planner/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/core/api/*.cc)

endif # not USE_FULL_TFLITE

ifeq (${USE_MEMRYX},1)
CFLAGS += -Iedge-impulse-sdk/third_party/gemmlowp
LDFLAGS += -Wl,--no-as-needed -ldl -ltensorflow-lite -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lcpuinfo -lpthreadpool -lpthread
ifeq (${TARGET_LINUX_AARCH64},1)
$(error MemryX drivers and runtime do not support AARCH64)
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

ifeq (${TARGET_JETSON_COMMON},1)
TENSORRT_VERSION ?=8
$(info TENSORRT_VERSION is ${TENSORRT_VERSION})
ifeq (${TENSORRT_VERSION},8.5.2)
TRT_LDFLAGS += -lei_debug -Ltflite/linux-jetson-nano/trt8.5.2/
else ifeq (${TENSORRT_VERSION},8)
TRT_LDFLAGS += -lei_debug -Ltflite/linux-jetson-nano/trt8/
else ifeq (${TENSORRT_VERSION},7)
TRT_LDFLAGS += -lei_debug7 -Ltflite/linux-jetson-nano/trt7/
else
$(error Invalid TensorRT version - supported versions are 7 and 8.)
endif # TENSORRT_VERSION
TRT_LDFLAGS += -lcudart -lnvinfer -lnvonnxparser
LDFLAGS += $(TRT_LDFLAGS) -lstdc++fs -Ltflite/linux-jetson-nano/ -Wl,--warn-unresolved-symbols,--unresolved-symbols=ignore-in-shared-libs
endif # TARGET_JETSON_COMMON

ifeq (${APP_CUSTOM},1)
NAME = custom
CXXSOURCES += source/custom.cpp
else ifeq (${APP_AUDIO},1)
NAME = audio
CXXSOURCES += source/audio.cpp
LDFLAGS += -lasound
else ifeq (${APP_CAMERA},1)
NAME = camera
CFLAGS += -Iopencv/build_opencv/ -Iopencv/opencv/include -Iopencv/opencv/3rdparty/include -Iopencv/opencv/3rdparty/quirc/include -Iopencv/opencv/3rdparty/carotene/include -Iopencv/opencv/3rdparty/ittnotify/include -Iopencv/opencv/3rdparty/openvx/include -Iopencv/opencv/modules/video/include -Iopencv/opencv/modules/flann/include -Iopencv/opencv/modules/core/include -Iopencv/opencv/modules/stitching/include -Iopencv/opencv/modules/imgproc/include -Iopencv/opencv/modules/objdetect/include -Iopencv/opencv/modules/gapi/include -Iopencv/opencv/modules/world/include -Iopencv/opencv/modules/ml/include -Iopencv/opencv/modules/imgcodecs/include -Iopencv/opencv/modules/dnn/include -Iopencv/opencv/modules/dnn/src/vkcom/include -Iopencv/opencv/modules/dnn/src/ocl4dnn/include -Iopencv/opencv/modules/dnn/src/tengine4dnn/include -Iopencv/opencv/modules/videoio/include -Iopencv/opencv/modules/highgui/include -Iopencv/opencv/modules/features2d/include -Iopencv/opencv/modules/ts/include -Iopencv/opencv/modules/photo/include -Iopencv/opencv/modules/calib3d/include
CXXSOURCES += source/camera.cpp
ifeq ($(UNAME_S),Linux) # on Linux set the library paths as well
LDFLAGS += -L/usr/local/lib -Wl,-R/usr/local/lib
endif
LDFLAGS += -lopencv_ml -lopencv_objdetect -lopencv_stitching  -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core
else ifeq (${APP_COLLECT},1)
NAME = collect
CXXSOURCES += source/collect.cpp
CSOURCES += $(wildcard ingestion-sdk-c/QCBOR/src/*.c) $(wildcard ingestion-sdk-c/mbedtls/library/*.c)
CFLAGS += -Iingestion-sdk-c/mbedtls/include -Iingestion-sdk-c/mbedtls/crypto/include -Iingestion-sdk-c/QCBOR/inc -Iingestion-sdk-c/QCBOR/src -Iingestion-sdk-c/inc -Iingestion-sdk-c/inc/signing
else ifeq (${APP_EIM},1)
NAME = model.eim
CXXSOURCES += source/eim.cpp
CFLAGS += -Ithird_party/
else
$(error Missing application, should have either APP_CUSTOM=1, APP_AUDIO=1, APP_CAMERA=1, APP_COLLECT=1 or APP_EIM=1)
endif

COBJECTS := $(patsubst %.c,%.o,$(CSOURCES))
CXXOBJECTS := $(patsubst %.cpp,%.o,$(CXXSOURCES))
CCOBJECTS := $(patsubst %.cc,%.o,$(CCSOURCES))

all: runner

.PHONY: runner clean

$(COBJECTS) : %.o : %.c
$(CXXOBJECTS) : %.o : %.cpp
$(CCOBJECTS) : %.o : %.cc

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.o: %.cc
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c $^ -o $@

runner: $(COBJECTS) $(CXXOBJECTS) $(CCOBJECTS)
	mkdir -p build
	$(CXX) $(COBJECTS) $(CXXOBJECTS) $(CCOBJECTS) -o build/$(NAME) $(LDFLAGS)

clean:
	rm -f $(COBJECTS)
	rm -f $(CCOBJECTS)
	rm -f $(CXXOBJECTS)

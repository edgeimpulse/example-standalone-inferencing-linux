NAME = linux-impulse-runner

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
CFLAGS += -DNDEBUG -DINCBIN_SILENCE_BITCODE_WARNING -DSILENCE_EI_CLASSFIER_OBJECT_DETECTION_COUNT_WARNING=1
CFLAGS += -g
ifeq (${CC}, clang)
    CFLAGS += -Wno-asm-operand-widths
endif
CXXFLAGS += -std=c++17
LDFLAGS += -lm -lstdc++
# tflite version may be overridden by included mk files, but default to 2.19.0
TFLITE_VERSION = 2.19.0

CSOURCES =	$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/TransformFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/CommonTables/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/BasicMathFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/ComplexMathFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/FastMathFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/SupportFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/MatrixFunctions/*.c) \
			$(wildcard edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/*.c)

CXXSOURCES =	$(wildcard tflite-model/*.cpp) \
				$(wildcard edge-impulse-sdk/dsp/kissfft/*.cpp) \
				$(wildcard edge-impulse-sdk/dsp/dct/*.cpp) \
				$(wildcard ./edge-impulse-sdk/dsp/memory.cpp) \
				$(wildcard edge-impulse-sdk/porting/posix/*.c*) \
				$(wildcard edge-impulse-sdk/porting/mingw32/*.c*) \
				$(wildcard third_party/base64/*.cpp) \
				$(wildcard third_party/jpeg/*.cpp)

CCSOURCES =

#######################################
# Include mk files
#######################################
include mks/ethos_linux.mk
include mks/renesas.mk
include mks/ti.mk
include mks/nvidia.mk
include mks/qualcomm.mk
include mks/memryx.mk
include mks/tflite.mk
include mks/akida.mk
include mks/tflm.mk

ifeq (${LINK_TFLITE_FLEX_LIBRARY},1)
    LDFLAGS += -ltensorflowlite_flex_${TFLITE_VERSION}
endif

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
ifeq ($(UNAME_S),Darwin) # add rpath on MacOS
LDFLAGS += -Wl,-rpath,/usr/local/lib
endif
LDFLAGS += -lopencv_ml -lopencv_objdetect -lopencv_stitching  -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core
else ifeq (${APP_VIDEO},1)
NAME = video
CFLAGS += -Iopencv/build_opencv/ -Iopencv/opencv/include -Iopencv/opencv/3rdparty/include -Iopencv/opencv/3rdparty/quirc/include -Iopencv/opencv/3rdparty/carotene/include -Iopencv/opencv/3rdparty/ittnotify/include -Iopencv/opencv/3rdparty/openvx/include -Iopencv/opencv/modules/video/include -Iopencv/opencv/modules/flann/include -Iopencv/opencv/modules/core/include -Iopencv/opencv/modules/stitching/include -Iopencv/opencv/modules/imgproc/include -Iopencv/opencv/modules/objdetect/include -Iopencv/opencv/modules/gapi/include -Iopencv/opencv/modules/world/include -Iopencv/opencv/modules/ml/include -Iopencv/opencv/modules/imgcodecs/include -Iopencv/opencv/modules/dnn/include -Iopencv/opencv/modules/dnn/src/vkcom/include -Iopencv/opencv/modules/dnn/src/ocl4dnn/include -Iopencv/opencv/modules/dnn/src/tengine4dnn/include -Iopencv/opencv/modules/videoio/include -Iopencv/opencv/modules/highgui/include -Iopencv/opencv/modules/features2d/include -Iopencv/opencv/modules/ts/include -Iopencv/opencv/modules/photo/include -Iopencv/opencv/modules/calib3d/include
CXXSOURCES += source/video.cpp
ifeq ($(UNAME_S),Linux) # on Linux set the library paths as well
LDFLAGS += -L/usr/local/lib -Wl,-R/usr/local/lib
endif
ifeq ($(UNAME_S),Darwin) # add rpath on MacOS
LDFLAGS += -Wl,-rpath,/usr/local/lib
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

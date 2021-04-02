EI_SDK?=edge-impulse-sdk

CFLAGS +=  -Wall -g -Wno-strict-aliasing
CFLAGS += -I.
CFLAGS += -Isource
CFLAGS += -Imodel-parameters
CFLAGS += -Itflite-model
CFLAGS += -Ithird_party/
CFLAGS += -Os
CFLAGS += -DNDEBUG
CFLAGS += -g
CXXFLAGS += -std=c++14
LDFLAGS += -lm -lstdc++

CSOURCES = $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/TransformFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/CommonTables/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/BasicMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/ComplexMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/FastMathFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/SupportFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/MatrixFunctions/*.c) $(wildcard edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/*.c)
CXXSOURCES = $(wildcard tflite-model/*.cpp) $(wildcard edge-impulse-sdk/dsp/kissfft/*.cpp) $(wildcard edge-impulse-sdk/dsp/dct/*.cpp) $(wildcard ./edge-impulse-sdk/dsp/memory.cpp) $(wildcard edge-impulse-sdk/porting/posix/*.c*) $(wildcard edge-impulse-sdk/porting/mingw32/*.c*)
CCSOURCES =

ifeq (${USE_FULL_TFLITE},1)
CFLAGS += -DEI_CLASSIFIER_USE_FULL_TFLITE=1
CFLAGS += -Itensorflow-lite/
ifeq (${TARGET_LINUX_ARMV7},1)
LDFLAGS += -L./tflite/linux-armv7 -ldl -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthread
endif
ifeq (${TARGET_LINUX_AARCH64},1)
CFLAGS += -Dfloat16_t=float
LDFLAGS += -L./tflite/linux-aarch64 -ldl -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthread
endif
ifeq (${TARGET_MAC_X86_64},1)
LDFLAGS += -L./tflite/mac-x86_64 -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthreadpool -lclog
endif
else
CFLAGS += -DTF_LITE_DISABLE_X86_NEON=1
CSOURCES += edge-impulse-sdk/tensorflow/lite/c/common.c
CCSOURCES += $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/kernels/internal/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/kernels/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/micro/memory_planner/*.cc) $(wildcard edge-impulse-sdk/tensorflow/lite/core/api/*.cc)
endif

ifeq (${APP_CUSTOM},1)
NAME = custom
CXXSOURCES += source/custom.cpp
else ifeq (${APP_AUDIO},1)
NAME = audio
CFLAGS += -lpthread
CXXSOURCES += source/audio.cpp
LDFLAGS += -lasound -lpthread
else
$(error Missing application, should have either APP_CUSTOM=1, APP_AUDIO=1 or APP_VISION=1)
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

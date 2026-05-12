#######################################
# USE_ETHOS
#######################################

ifeq (${USE_ETHOS},1)
CFLAGS += -DEI_ETHOS_LINUX
CFLAGS += -Iedge-impulse-sdk/third_party/ethos_kernel_driver/include/
CFLAGS += -Iedge-impulse-sdk/third_party/ethos_driver_library/include
CXXSOURCES += third_party/ethos-u-driver-stack-imx/driver_library/src/ethosu.cpp
LDFLAGS += -lrt
endif

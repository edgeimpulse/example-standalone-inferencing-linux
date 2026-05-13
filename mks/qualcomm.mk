#######################################
# USE_QUALCOMM_QNN
#######################################

ifeq (${USE_QUALCOMM_QNN},1)
ifndef QNN_SDK_ROOT
$(error QNN_SDK_ROOT is not set, install QNN Engine Direct and set it to the installation directory)
endif
USE_FULL_TFLITE=1
CFLAGS += -I${QNN_SDK_ROOT}/include
CFLAGS += -Iedge-impulse-sdk
CFLAGS += -DEI_CLASSIFIER_USE_QNN_DELEGATES
LDFLAGS += -L${QNN_SDK_ROOT}/lib/aarch64-ubuntu-gcc9.4 -lQnnTFLiteDelegate
endif

#include <stdio.h>

#if EI_CLASSIFIER_USE_FULL_TFLITE
#include <thread>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define ITERATION_COUNT         50
#define ITERATION_COUNT_SSD     10

#else
#include <cmath>
#include "edge-impulse-sdk/tensorflow/lite/micro/all_ops_resolver.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/micro_error_reporter.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/micro_interpreter.h"
#include "edge-impulse-sdk/tensorflow/lite/schema/schema_generated.h"
#include "edge-impulse-sdk/classifier/ei_aligned_malloc.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/micro_ops.h"

#define ITERATION_COUNT         10
#define ITERATION_COUNT_SSD     1

#define EI_TFLITE_RESOLVER static tflite::MicroMutableOpResolver<8> resolver; \
    resolver.AddFullyConnected(); \
    resolver.AddSoftmax(); \
    resolver.AddAdd(); \
    resolver.AddConv2D(); \
    resolver.AddDepthwiseConv2D(); \
    resolver.AddReshape(); \
    resolver.AddMaxPool2D(); \
    resolver.AddPad();

#endif

#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "benchmark-nn/gestures-large-f32/tflite-trained.h"
#include "benchmark-nn/gestures-large-i8/tflite-trained.h"
#include "benchmark-nn/image-32-32-mobilenet-f32/tflite-trained.h"
#include "benchmark-nn/image-32-32-mobilenet-i8/tflite-trained.h"
#include "benchmark-nn/image-96-96-mobilenet-f32/tflite-trained.h"
#include "benchmark-nn/image-96-96-mobilenet-i8/tflite-trained.h"
#include "benchmark-nn/image-320-320-mobilenet-ssd-f32/tflite-trained.h"
#include "benchmark-nn/keywords-2d-f32/tflite-trained.h"
#include "benchmark-nn/keywords-2d-i8/tflite-trained.h"

// You can toggle these on / off in case devices don't have enough flash to hold all of them in one go
// just concat the output afterwards
#define GESTURES_F32           1
#define GESTURES_I8            1
#define MOBILENET_32_32_F32    1
#define MOBILENET_32_32_I8     1
#define MOBILENET_96_96_F32    1
#define MOBILENET_96_96_I8     1
#define MOBILENET_320_320_F32  1
#define KEYWORDS_F32           1
#define KEYWORDS_I8            1

int run_model_tflite_full(const unsigned char *trained_tflite, size_t trained_tflite_len, int iterations, uint64_t *time_us) {
    std::unique_ptr<tflite::FlatBufferModel> model = nullptr;
    std::unique_ptr<tflite::Interpreter> interpreter = nullptr;
    if (!model) {
        model = tflite::FlatBufferModel::BuildFromBuffer((const char*)trained_tflite, trained_tflite_len);
        if (!model) {
            ei_printf("Failed to build TFLite model from buffer\n");
            return EI_IMPULSE_TFLITE_ERROR;
        }

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);

        if (!interpreter) {
            ei_printf("Failed to construct interpreter\n");
            return EI_IMPULSE_TFLITE_ERROR;
        }

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            ei_printf("AllocateTensors failed\n");
            return EI_IMPULSE_TFLITE_ERROR;
        }

        int hw_thread_count = (int)std::thread::hardware_concurrency();
        hw_thread_count -= 1; // leave one thread free for the other application
        if (hw_thread_count < 1) {
            hw_thread_count = 1;
        }

        if (interpreter->SetNumThreads(hw_thread_count) != kTfLiteOk) {
            ei_printf("SetNumThreads failed\n");
            return EI_IMPULSE_TFLITE_ERROR;
        }
    }

    auto start_us = ei_read_timer_us();

    for (int ix = 0; ix < iterations; ix++) {
        interpreter->Invoke();
    }

    auto end_us = ei_read_timer_us();

    *time_us = end_us - start_us;

    return 0;
}

int main() {
    std::map<const char*, int> res;

    #if GESTURES_F32
    {
        uint64_t time_us;
        const char *test_name = "gestures-large-f32";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_gestures_large_f32, trained_tflite_gestures_large_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if GESTURES_I8
    {
        uint64_t time_us;
        const char *test_name = "gestures-large-i8";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_gestures_large_i8, trained_tflite_gestures_large_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }
        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MOBILENET_32_32_F32
    {
        uint64_t time_us;
        const char *test_name = "image-32-32-mobilenet-f32";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_image_32_32_f32, trained_tflite_image_32_32_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MOBILENET_32_32_I8
    {
        uint64_t time_us;
        const char *test_name = "image-32-32-mobilenet-i8";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_image_32_32_i8, trained_tflite_image_32_32_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MOBILENET_96_96_F32
    {
        uint64_t time_us;
        const char *test_name = "image-96-96-mobilenet-f32";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_image_96_96_f32, trained_tflite_image_96_96_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MOBILENET_96_96_I8
    {
        uint64_t time_us;
        const char *test_name = "image-96-96-mobilenet-i8";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_image_96_96_i8, trained_tflite_image_96_96_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MOBILENET_320_320_F32
    {
        uint64_t time_us;
        const char *test_name = "image-320-320-mobilenet-ssd-f32";
        int iterations = ITERATION_COUNT_SSD;
        int x = run_model_tflite_full(trained_tflite_image_320_320_ssd_f32, trained_tflite_image_320_320_ssd_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if KEYWORDS_F32
    {
        uint64_t time_us;
        const char *test_name = "keywords-2d-f32";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_keywords_f32, trained_tflite_keywords_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if KEYWORDS_I8
    {
        uint64_t time_us;
        const char *test_name = "keywords-2d-i8";
        int iterations = ITERATION_COUNT;
        int x = run_model_tflite_full(trained_tflite_keywords_i8, trained_tflite_keywords_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name);
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    ei_printf("{\n");
    for (auto const& x : res) {
        ei_printf("    %s: %f,\n", x.first, ((float)x.second / 1000.0f));
    }
    ei_printf("}\n");

}

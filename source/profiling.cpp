#include <stdio.h>
#include <stdlib.h>
#include <map>

#if EI_CLASSIFIER_USE_FULL_TFLITE
#include <thread>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <filesystem>
namespace fs = std::filesystem;

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

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

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
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "edge-impulse-sdk/dsp/numpy_types.h"
#include "edge-impulse-sdk/classifier/ei_run_dsp.h"
#include "model-parameters/model_metadata.h"
#include "model-parameters/mfcc_input.h"
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
#define EI_CLASSIFIER_TFLITE_ARENA_SIZE     (80 * 1024)     // Update this to grab as much RAM as possible on embedded systems
#define GESTURES_F32           1
#define GESTURES_I8            1
#define MOBILENET_32_32_F32    1
#define MOBILENET_32_32_I8     1
#define MOBILENET_96_96_F32    1
#define MOBILENET_96_96_I8     1
#define MOBILENET_320_320_F32  1
#define KEYWORDS_F32           1
#define KEYWORDS_I8            1
#define MFCC                   1

#if EI_CLASSIFIER_USE_FULL_TFLITE
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
#else // TensorFlow Lite Micro
int run_model_tflite_full(const unsigned char *trained_tflite, size_t trained_tflite_len, int iterations, uint64_t *time_us) {
    uint8_t *tensor_arena = (uint8_t*)ei_aligned_calloc(16, EI_CLASSIFIER_TFLITE_ARENA_SIZE);
    if (tensor_arena == NULL) {
        ei_printf("Failed to allocate TFLite arena (%d bytes)\n", EI_CLASSIFIER_TFLITE_ARENA_SIZE);
        return EI_IMPULSE_TFLITE_ARENA_ALLOC_FAILED;
    }

    const tflite::Model* model = tflite::GetModel(trained_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        ei_aligned_free(tensor_arena);
        return EI_IMPULSE_TFLITE_ERROR;
    }

    EI_TFLITE_RESOLVER;

    tflite::MicroInterpreter *interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, EI_CLASSIFIER_TFLITE_ARENA_SIZE);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors(true);
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        ei_aligned_free(tensor_arena);
        return EI_IMPULSE_TFLITE_ERROR;
    }

    auto start_us = ei_read_timer_us();

    for (int ix = 0; ix < iterations; ix++) {
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed (%d)\n", invoke_status);
            ei_aligned_free(tensor_arena);
            return EI_IMPULSE_TFLITE_ERROR;
        }
    }

    auto end_us = ei_read_timer_us();

    *time_us = end_us - start_us;

    delete interpreter;
    ei_aligned_free(tensor_arena);

    return 0;
}
#endif

int main(int argc, char **argv) {
    std::map<std::string, int> res;

    int it_count = ITERATION_COUNT;
    int it_count_ssd = ITERATION_COUNT_SSD;

    #if EI_CLASSIFIER_USE_FULL_TFLITE
    if (argc > 1) {
        for (int ix = 1; ix < argc; ix++) {
            FILE *f = fopen(argv[ix], "rb");
            if (f == NULL) {
                ei_printf("ERR: Could not open file %s\n", argv[ix]);
                return 1;
            }

            fseek(f, 0, SEEK_END);
            auto lsize = ftell(f);
            rewind(f);

            int it_count_custom = it_count;

            // more than 1MB?
            if (lsize > 1 * 1024 * 1024) {
                it_count_custom = 10;
            }

            uint8_t *data = (uint8_t*)malloc(lsize);
            if (!data) {
                ei_printf("ERR: Could not allocate buffer for file\n");
                return 1;
            }
            fread(data, 100 * 1024 * 1024, 1, f);
            fclose(f);

            auto test_name = fs::path(argv[ix]).filename();

            uint64_t time_us;
            int iterations = it_count_custom;
            int x = run_model_tflite_full((const unsigned char *)data, lsize, iterations, &time_us);
            if (x != 0) {
                ei_printf("ERR: Failed to run test (%d)\n", x);
                return 1;
            }

            ei_printf("Test: %s\n", test_name.c_str());
            ei_printf("Iterations: %d\n", iterations);
            ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
            ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
            ei_printf("\n");
            res[test_name] = (int)(time_us / iterations);

            free(data);
        }

        // time matters here (as we most likely run this from a job), so just run few iterations now
        it_count = 5;
        it_count_ssd = 3;
    }
    #endif

    #if GESTURES_F32
    {
        uint64_t time_us;
        std::string test_name = "gestures-large-f32";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_gestures_large_f32, trained_tflite_gestures_large_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "gestures-large-i8";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_gestures_large_i8, trained_tflite_gestures_large_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }
        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "image-32-32-mobilenet-f32";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_image_32_32_f32, trained_tflite_image_32_32_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "image-32-32-mobilenet-i8";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_image_32_32_i8, trained_tflite_image_32_32_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "image-96-96-mobilenet-f32";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_image_96_96_f32, trained_tflite_image_96_96_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "image-96-96-mobilenet-i8";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_image_96_96_i8, trained_tflite_image_96_96_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "image-320-320-mobilenet-ssd-f32";
        int iterations = it_count_ssd;
        int x = run_model_tflite_full(trained_tflite_image_320_320_ssd_f32, trained_tflite_image_320_320_ssd_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "keywords-2d-f32";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_keywords_f32, trained_tflite_keywords_f32_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
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
        std::string test_name = "keywords-2d-i8";
        int iterations = it_count;
        int x = run_model_tflite_full(trained_tflite_keywords_i8, trained_tflite_keywords_i8_len, iterations, &time_us);
        if (x != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", x);
            return 1;
        }

        ei_printf("Test: %s\n", test_name.c_str());
        ei_printf("Iterations: %d\n", iterations);
        ei_printf("Total time: %d ms.\n", (int)(time_us / 1000));
        ei_printf("Time per inference: %d us.\n", (int)(time_us / iterations));
        ei_printf("\n");
        res[test_name] = (int)(time_us / iterations);
    }
    #endif

    #if MFCC
    {
        const ei_dsp_config_mfcc_t config = {
            2, // uint32_t blockId
            1, // int implementationVersion
            1, // int length of axes
            13, // int num_cepstral
            0.02f, // float frame_length
            0.01f, // float frame_stride
            32, // int num_filters
            256, // int fft_length
            101, // int win_size
            300, // int low_frequency
            0, // int high_frequency
            0.98f, // float pre_cof
            1 // int pre_shift
        };
        const float frequency = 16000.0f;

        int (*extract_fn)(ei::signal_t *signal, ei::matrix_t *output_matrix, void *config, const float frequency) = extract_mfcc_features;

        ei::signal_t signal;
        int ret;
        ei::matrix_t output_matrix(1, 99*13);
        ret = numpy::signal_from_buffer(mfcc_input, 16000, &signal);
        if (ret != 0) {
            ei_printf("ERR: Failed to run test (%d)\n", ret);
            return 1;
        }

        int iterations = it_count;

        auto start_us = ei_read_timer_us();

        for (int ix = 0; ix < iterations; ix++) {
            ret = extract_fn(&signal, &output_matrix, (void*)&config, frequency);
            if (ret != 0) {
                ei_printf("ERR: Failed to run test (%d)\n", ret);
                return 1;
            }
        }

        auto end_us = ei_read_timer_us();

        auto time_us = end_us - start_us;

        int64_t time_per_inference_us = (int64_t)(time_us / iterations);

        // Cortex-M4F @ 80MHz takes 456000 us. and that's 59 mips
        // so we can calculate back...
        float factor = (float)time_per_inference_us / 456000.0f;
        int dsp_mips = (int)(59.0f / factor);

        ei_printf("DSP MIPS is %d\n\n", dsp_mips);
    }
    #endif

    ei_printf("{\n");
    for (auto x = res.begin(); x != res.end(); x++) {
        ei_printf("    \"%s\": %f", x->first.c_str(), ((float)x->second / 1000.0f));
        if (std::next(x) != res.end()) {
            ei_printf(",");
        }
        ei_printf("\n");
    }
    ei_printf("}\n");

}

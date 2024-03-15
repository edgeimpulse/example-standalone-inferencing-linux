#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <filesystem>
#include <stdlib.h>
#include "tflite/linux-jetson-nano/libeitrt.h"
#include <map>

#if __APPLE__
#include <mach-o/dyld.h>
#else
#include <linux/limits.h>
#endif

EiTrt *ei_trt_handle = NULL;

inline bool file_exists(char *model_file_name)
{
    if (FILE *file = fopen(model_file_name, "r")) {
        fclose(file);
        return true;
    }
    else {
        return false;
    }
}

#define ITERATION_COUNT         50
#define ITERATION_COUNT_SSD     10

#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "edge-impulse-sdk/dsp/numpy_types.h"
#include "edge-impulse-sdk/classifier/ei_run_dsp.h"
#include "model-parameters/model_metadata.h"
#include "model-parameters/mfcc_input.h"
#include "benchmark-nn/gestures-large-f32/onnx-trained.h"
//#include "benchmark-nn/gestures-large-i8/onnx-trained.h"
#include "benchmark-nn/image-32-32-mobilenet-f32/onnx-trained.h"
//#include "benchmark-nn/image-32-32-mobilenet-i8/onnx-trained.h"
#include "benchmark-nn/image-96-96-mobilenet-f32/onnx-trained.h"
//#include "benchmark-nn/image-96-96-mobilenet-i8/onnx-trained.h"
//#include "benchmark-nn/image-320-320-mobilenet-ssd-f32/onnx-trained.h"
#include "benchmark-nn/keywords-2d-f32/onnx-trained.h"
//#include "benchmark-nn/keywords-2d-i8/onnx-trained.h"

// You can toggle these on / off in case devices don't have enough flash to hold all of them in one go
// just concat the output afterwards
#define INPUT_OUTPUT_BUF_SIZE     (80 * 1024)     // Update this to grab as much RAM as possible on embedded systems
#define GESTURES_F32           1
#define GESTURES_I8            0
#define MOBILENET_32_32_F32    1
#define MOBILENET_32_32_I8     0
#define MOBILENET_96_96_F32    1
#define MOBILENET_96_96_I8     0
#define MOBILENET_320_320_F32  0
#define KEYWORDS_F32           1
#define KEYWORDS_I8            0
#define MFCC                   1

int run_model_tensorrt(const unsigned char *trained_onnx, size_t trained_onnx_len, const char* hash, int iterations, uint64_t *time_us) {

    static char current_exe_path[PATH_MAX] = { 0 };

#if __APPLE__
    uint32_t len = PATH_MAX;
    if (_NSGetExecutablePath(current_exe_path, &len) != 0) {
        current_exe_path[0] = '\0'; // buffer too small
    }
    else {
        // resolve symlinks, ., .. if possible
        char *canonical_path = realpath(current_exe_path, NULL);
        if (canonical_path != NULL)
        {
            strncpy(current_exe_path, canonical_path, len);
            free(canonical_path);
        }
    }
#else
    int readlink_res = readlink("/proc/self/exe", current_exe_path, PATH_MAX);
    if (readlink_res < 0) {
        printf("readlink_res = %d\n", readlink_res);
        current_exe_path[0] = '\0'; // failed to find location
    }
#endif

    static char model_file_name[PATH_MAX];

    if (strlen(current_exe_path) == 0) {
        // could not determine current exe path, use /tmp for the engine file
        snprintf(
            model_file_name,
            PATH_MAX,
            "/tmp/ei-%s.engine",
            hash);
    }
    else {
        std::filesystem::path p(current_exe_path);
        snprintf(
            model_file_name,
            PATH_MAX,
            "%s/%s-project-%s.engine",
            p.parent_path().c_str(),
            p.stem().c_str(),
            hash);
    }

    bool fexists = file_exists(model_file_name);
    if (!fexists) {
        ei_printf("INFO: Model file '%s' does not exist, creating...\n", model_file_name);

        FILE *file = fopen(model_file_name, "w");
        if (!file) {
            ei_printf("ERR: TensorRT init failed to open '%s'\n", model_file_name);
            return EI_IMPULSE_TENSORRT_INIT_FAILED;
        }

        if (fwrite(trained_onnx, trained_onnx_len, 1, file) != 1) {
            ei_printf("ERR: TensorRT init fwrite failed.\n");
            return EI_IMPULSE_TENSORRT_INIT_FAILED;
        }

        if (fclose(file) != 0) {
            ei_printf("ERR: TensorRT init fclose failed.\n");
            return EI_IMPULSE_TENSORRT_INIT_FAILED;
        }
    }

    uint32_t out_data_size = INPUT_OUTPUT_BUF_SIZE;

    float *out_data = (float*)ei_malloc(out_data_size * sizeof(float));
    if (out_data == nullptr) {
        ei_printf("ERR: Cannot allocate memory for output data \n");
    }

    // lazy initialize tensorRT context
    if (ei_trt_handle == nullptr) {
        ei_trt_handle = libeitrt::create_EiTrt(model_file_name, false);
    }

    uint32_t in_data_size = INPUT_OUTPUT_BUF_SIZE;
    float *in_data = (float*)ei_malloc(in_data_size * sizeof(float));
    if (in_data == nullptr) {
        ei_printf("ERR: Cannot allocate memory for input data \n");
    }

    // get the first loading of the model out of the way
    auto load_start_us = ei_read_timer_us();
    libeitrt::infer(ei_trt_handle, in_data, out_data, out_data_size);
    auto load_end_us = ei_read_timer_us();
    ei_printf("load time took %d\n", (int) (load_end_us - load_start_us));

    auto start_us = ei_read_timer_us();

    for (int ix = 0; ix < iterations; ix++) {
        libeitrt::infer(ei_trt_handle, in_data, out_data, out_data_size);
    }

    auto end_us = ei_read_timer_us();

    *time_us = end_us - start_us;

    return 0;
}

int main(int argc, char **argv) {
    std::map<std::string, int> res;

    int it_count = ITERATION_COUNT;
    //int it_count_ssd = ITERATION_COUNT_SSD;

    #if GESTURES_F32
    {
        uint64_t time_us;
        std::string test_name = "gestures-large-f32";
        int iterations = it_count;
        int x = run_model_tensorrt(trained_onnx_gestures_large_f32, trained_onnx_gestures_large_f32_len, trained_onnx_gestures_large_f32_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_gestures_large_i8, trained_onnx_gestures_large_i8_len, trained_onnx_gestures_large_i8_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_image_32_32_f32, trained_onnx_image_32_32_f32_len, trained_onnx_image_32_32_f32_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_image_32_32_i8, trained_onnx_image_32_32_i8_len, trained_onnx_image_32_32_i8_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_image_96_96_f32, trained_onnx_image_96_96_f32_len, trained_onnx_image_96_96_f32_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_image_96_96_i8, trained_onnx_image_96_96_i8_len, trained_onnx_image_96_96_i8_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_image_320_320_ssd_f32, trained_onnx_image_320_320_ssd_f32_len, trained_onnx_image_320_320_ssd_f32_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_keywords_f32, trained_onnx_keywords_f32_len, trained_onnx_keywords_f32_hash, iterations, &time_us);
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
        int x = run_model_tensorrt(trained_onnx_keywords_i8, trained_onnx_keywords_i8_len, trained_onnx_keywords_i8_hash, iterations, &time_us);
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

#ifndef _POSE_ESTIMATION_DSP_BLOCK_H_
#define _POSE_ESTIMATION_DSP_BLOCK_H_

#include <thread>
// #include "tensorflow-lite/tensorflow/lite/c/common.h"
// #include "tensorflow-lite/tensorflow/lite/interpreter.h"
// #include "tensorflow-lite/tensorflow/lite/kernels/register.h"
// #include "tensorflow-lite/tensorflow/lite/model.h"
// #include "tensorflow-lite/tensorflow/lite/optional_debug_tools.h"
#include "model.h"

int extract_pose_estimation_features(signal_t *signal, matrix_t *output_matrix, void *config_ptr, const float frequency) {
    signal->get_data(0, 51, output_matrix->buffer);
    return 0;

    // printf("0\n");

    // static std::unique_ptr<tflite::FlatBufferModel> model = nullptr;
    // static std::unique_ptr<tflite::Interpreter> interpreter = nullptr;

    // printf("1\n");

    // if (!model) {
    //     model = tflite::FlatBufferModel::BuildFromBuffer((const char*)pose_estimation_tflite, pose_estimation_tflite_length);
    //     if (!model) {
    //         ei_printf("Failed to build TFLite model from buffer\n");
    //         return EI_IMPULSE_TFLITE_ERROR;
    //     }

    //     printf("2\n");

    //     tflite::ops::builtin::BuiltinOpResolver resolver;
    //     tflite::InterpreterBuilder builder(*model, resolver);
    //     builder(&interpreter);

    //     printf("3\n");

    //     if (!interpreter) {
    //         ei_printf("Failed to construct interpreter\n");
    //         return EI_IMPULSE_TFLITE_ERROR;
    //     }

    //     if (interpreter->AllocateTensors() != kTfLiteOk) {
    //         ei_printf("AllocateTensors failed\n");
    //         return EI_IMPULSE_TFLITE_ERROR;
    //     }

    //     int hw_thread_count = (int)std::thread::hardware_concurrency();
    //     hw_thread_count -= 1; // leave one thread free for the other application
    //     if (hw_thread_count < 1) {
    //         hw_thread_count = 1;
    //     }

    //     if (interpreter->SetNumThreads(hw_thread_count) != kTfLiteOk) {
    //         ei_printf("SetNumThreads failed\n");
    //         return EI_IMPULSE_TFLITE_ERROR;
    //     }
    // }

    // ei_printf("signal total_length %d\n", signal->total_length);

    // float* input = interpreter->typed_input_tensor<float>(0);

    // if (!input) {
    //     return EI_IMPULSE_INPUT_TENSOR_WAS_NULL;
    // }

    // int x = signal->get_data(0, signal->total_length, input);
    // ei_printf("get data returned %d\n", x);

    // uint64_t ctx_start_us = ei_read_timer_us();

    // interpreter->Invoke();

    // uint64_t ctx_end_us = ei_read_timer_us();

    // float* out_data = interpreter->typed_output_tensor<float>(EI_CLASSIFIER_TFLITE_OUTPUT_DATA_TENSOR);

    // if (!out_data) {
    //     return EI_IMPULSE_OUTPUT_TENSOR_WAS_NULL;
    // }

    // memcpy(output_matrix->buffer, out_data, 51 * 4);

    // ei_printf("DSP Predictions (time: %d ms.):\n", ctx_end_us - ctx_start_us);

    // return 0;
}

#endif // _POSE_ESTIMATION_DSP_BLOCK_H_
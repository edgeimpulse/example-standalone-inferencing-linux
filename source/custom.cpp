/* Edge Impulse Linux SDK
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/inferencing_engines/tflite_full.h"
#include "bitmap_helper.h"
#include "yolov5-part2/drpai_out.h"
#include "yolov5-part2/model-part2.h"

int main(int argc, char **argv) {

    static std::unique_ptr<tflite::FlatBufferModel> model = nullptr;
    static std::unique_ptr<tflite::Interpreter> interpreter = nullptr;

    if (!model) {
        model = tflite::FlatBufferModel::BuildFromBuffer((const char*)yolov5_part2, yolov5_part2_len);
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

    const size_t drpai_features = drpai_buff_size;

    const size_t els_per_grid = drpai_features / ((NUM_GRID_1 * NUM_GRID_1) + (NUM_GRID_2 * NUM_GRID_2) + (NUM_GRID_3 * NUM_GRID_3));

    const size_t grid_1_offset = 0;
    const size_t grid_1_size = (NUM_GRID_1 * NUM_GRID_1) * els_per_grid;

    const size_t grid_2_offset = grid_1_offset + grid_1_size;
    const size_t grid_2_size = (NUM_GRID_2 * NUM_GRID_2) * els_per_grid;

    const size_t grid_3_offset = grid_2_offset + grid_2_size;
    const size_t grid_3_size = (NUM_GRID_3 * NUM_GRID_3) * els_per_grid;

    // Now we don't know the exact tensor order for some reason
    // so let's do that dynamically
    for (size_t ix = 0; ix < 3; ix++) {
        TfLiteTensor * tensor = interpreter->input_tensor(ix);
        size_t tensor_size = 1;
        for (size_t ix = 0; ix < tensor->dims->size; ix++) {
            tensor_size *= tensor->dims->data[ix];
        }

        printf("input tensor %d, tensor_size=%d\n", (int)ix, (int)tensor_size);

        float *input = interpreter->typed_input_tensor<float>(ix);

        if (tensor_size == grid_1_size) {
            memcpy(input, drpai_buff + grid_1_offset, grid_1_size * sizeof(float));
        }
        else if (tensor_size == grid_2_size) {
            memcpy(input, drpai_buff + grid_2_offset, grid_2_size * sizeof(float));
        }
        else if (tensor_size == grid_3_size) {
            memcpy(input, drpai_buff + grid_3_offset, grid_3_size * sizeof(float));
        }
        else {
            printf("Cannot determine which grid to use for input tensor %d with %d tensor size\n",
                (int)ix, (int)tensor_size);
            return 1;
        }
    }

    uint64_t ctx_start_us = ei_read_timer_us();

    interpreter->Invoke();

    uint64_t ctx_end_us = ei_read_timer_us();

    printf("Invoke took %d ms.\n", (int)((ctx_end_us - ctx_start_us) / 1000));

    float* out_data = interpreter->typed_output_tensor<float>(0);

    const size_t out_size = 1 * 9072 * 7;

    printf("First 20 bytes: ");
    for (size_t ix = 0; ix < 20; ix++) {
        printf("%f ", out_data[ix]);
    }
    printf("\n");

    // printf("Last 5 bytes: ");
    // for (size_t ix = out_size - 5; ix < out_size; ix++) {
    //     printf("%f ", out_data[ix]);
    // }
    // printf("\n");

    ei_impulse_t impulse = { 0 };
    impulse.input_width = IMAGE_WIDTH;
    impulse.input_height = IMAGE_HEIGHT;
    impulse.label_count = NUM_CLASS;
    impulse.object_detection_threshold = 0.3f;

    ei_impulse_result_t result;

    fill_result_struct_f32_yolov5((const ei_impulse_t *)&impulse, &result, 5, out_data, out_size);

    printf("\n\nC++ results:\nbounding_boxes_count = %d\n", (int)result.bounding_boxes_count);
    for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
        ei_impulse_result_bounding_box_t *r = &result.bounding_boxes[ix];
        printf("    %s %.5f [ %d, %d, %d, %d ]\n",
            r->label, r->value, r->x, r->y, r->width, r->height);
    }
    printf("\n");
}

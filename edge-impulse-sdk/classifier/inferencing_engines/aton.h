/*
 * Copyright (c) 2024 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

 #ifndef _EI_CLASSIFIER_INFERENCING_ENGINE_ATON_H

#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_ATON)

/* Include ----------------------------------------------------------------- */
#include "edge-impulse-sdk/tensorflow/lite/kernels/custom/tree_ensemble_classifier.h"
#include "edge-impulse-sdk/classifier/ei_fill_result_struct.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/classifier/ei_run_dsp.h"
#include "edge-impulse-sdk/porting/ei_logging.h"

#include "ll_aton_runtime.h"
#include "app_config.h"

/* Private variables ------------------------------------------------------- */
static uint8_t *nn_in;
static uint8_t *nn_out;

static const LL_Buffer_InfoTypeDef *nn_in_info;
static const LL_Buffer_InfoTypeDef *nn_out_info;

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default);


EI_IMPULSE_ERROR run_nn_inference_image_quantized(
    const ei_impulse_t *impulse,
    signal_t *signal,
    ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{
    EI_IMPULSE_ERROR fill_res = EI_IMPULSE_OK;
    extern uint8_t *global_camera_buffer;
    extern uint8_t *snapshot_buf;
    // this needs to be changed for multi-model, multi-impulse
    static bool first_run = true;

    uint64_t ctx_start_us = ei_read_timer_us();

    #if DATA_OUT_FORMAT_FLOAT32
    static float32_t *nn_out;
    #else
    static uint8_t *nn_out;
    #endif
    static uint32_t nn_out_len;

    if(first_run == true) {

        nn_in_info = LL_ATON_Input_Buffers_Info_Default();
        nn_out_info = LL_ATON_Output_Buffers_Info_Default();

        nn_in = (uint8_t *) nn_in_info[0].addr_start.p;
        uint32_t nn_in_len = LL_Buffer_len(&nn_in_info[0]);
        nn_out = (uint8_t *) nn_out_info[0].addr_start.p;


        #if DATA_OUT_FORMAT_FLOAT32
        nn_out = (float32_t *) nn_out_info[0].addr_start.p;
        #else
        nn_out = (uint8_t *) nn_out_info[0].addr_start.p;
        #endif
        nn_out_len = LL_Buffer_len(&nn_out_info[0]);

        first_run = false;
    }

    memcpy(nn_in, snapshot_buf, impulse->input_width * impulse->input_height * 3);
    #ifdef USE_DCACHE
    SCB_CleanInvalidateDCache_by_Addr(nn_in, impulse->input_width * impulse->input_height * 3);
    #endif

    LL_ATON_RT_Main(&NN_Instance_Default);

    #ifdef USE_DCACHE
    SCB_CleanInvalidateDCache_by_Addr(nn_out, nn_out_len);
    #endif

    ei_learning_block_config_tflite_graph_t *block_config = (ei_learning_block_config_tflite_graph_t *)impulse->learning_blocks[0].config;
    if (block_config->classification_mode == EI_CLASSIFIER_CLASSIFICATION_MODE_OBJECT_DETECTION) {
        switch (block_config->object_detection_last_layer) {

            case EI_CLASSIFIER_LAST_LAYER_YOLOV5:
                #if MODEL_OUTPUT_IS_FLOAT
                fill_res = fill_result_struct_f32_yolov5(
                    ei_default_impulse.impulse,
                    &result,
                    6, // hard coded for now
                    (float *)&data,//output.data.uint8,
                    // output.params.zero_point,
                    // output.params.scale,
                    ei_default_impulse.impulse->tflite_output_features_count);
                #else
                fill_res = fill_result_struct_quantized_yolov5(
                    impulse,        
                    block_config,
                    result,
                    6, // hard coded for now
                    (uint8_t *)nn_out,
                    nn_out_info[0].offset[0],
                    nn_out_info[0].scale[0],
                    nn_out_len);
                #endif
                break;

            case EI_CLASSIFIER_LAST_LAYER_FOMO:
                fill_res = fill_result_struct_i8_fomo(
                    impulse,
                    block_config,
                    result,
                    (int8_t *)nn_out,
                    nn_out_info[0].offset[0],
                    nn_out_info[0].scale[0],
                    impulse->fomo_output_size,
                    impulse->fomo_output_size);
                break;
        
            default:
                ei_printf("ERR: Unsupported object detection last layer (%d)\n",
                    block_config->object_detection_last_layer);
                fill_res = EI_IMPULSE_UNSUPPORTED_INFERENCING_ENGINE;
                break;
        }

    }
    // if we copy the output, we don't need to process it as classification
    else
    {
        if (!result->copy_output) {
            bool int8_output = 1; //quantized hardcoded for now
            if (int8_output) {
                fill_res = fill_result_struct_i8(impulse, result, (int8_t *)nn_out, nn_out_info[0].offset[0], nn_out_info[0].scale[0], debug);
            }
            else {
                fill_res = fill_result_struct_f32(impulse, result,(float *)nn_out, debug);
            }
        }
    }

    result->timing.classification_us = ei_read_timer_us() - ctx_start_us;

    return fill_res;
}


/**
 * @brief      Do neural network inferencing over the processed feature matrix
 *
 * @param      fmatrix  Processed matrix
 * @param      result   Output classifier results
 * @param[in]  debug    Debug output enable
 *
 * @return     The ei impulse error.
 */
EI_IMPULSE_ERROR run_nn_inference(
    const ei_impulse_t *impulse,
    ei_feature_t *fmatrix,
    uint32_t learn_block_index,
    uint32_t* input_block_ids,
    uint32_t input_block_ids_size,
    ei_impulse_result_t *result,
    void *config_ptr,
    bool debug = false)
{


    return EI_IMPULSE_OK;
}

#endif // EI_CLASSIFIER_INFERENCING_ENGINE
#endif // _EI_CLASSIFIER_INFERENCING_ENGINE_ATON_H
/* The Clear BSD License
 *
 * Copyright (c) 2025 EdgeImpulse Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <memory>

//forward declare the wrapper class side TensorRT so we don't bring in all the dependencies
class EiTrt;

namespace libeitrt
{

/**
 * @brief Creates and initializes a context for building and running TensorRT models.
 *
 * The models genenerated (or managed) from this context is then persisted via the EiTrt
 * object until it is deleted,  to provide for fastest inference with lowest
 * overhead.
 * 
 * WARNING: This function leaks..the handle can not be deleted b/c of forward declaration
 * The fix for this is to define an interface (virtual class) that has a virtual destructor
 * And also the infer function (although this way is more C friendly!)
 * My bad...should have done that from get go.
 * 
 * @param debug enable debug if true, disable otherwise.
 * @return std::unique_ptr<EiTrt> EiTrt handle.  Contained ptr is NULL if error
 */
EiTrt* create_EiTrt(bool debug);

/**
 * @brief Builds and initializes an inference engine for TensorRT.
 * If the engine has already been created from the provided file path, then
 * the engine is loaded from disk.
 *
 * The engine is then persisted via the EiTrt object until it is deleted,
 * to provide for fastest inference with lowest overhead
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id an index to associate with the model.
 * @param model_file_name Model file path.
 * Should have hash appended so that engines are regenerated when models change!
 * @return true if building (or loading) the TensorRT model was successful.
 */
bool build(EiTrt* ei_trt_handle, int model_id, const char *model_file_name);

/**
 *  @brief Warms up the model on the GPU for given warm_up_ms ms.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @param warm_up_ms the duration to loop and run inference.
 * @return true if warming up the model was successful.
 */
bool warmUp(EiTrt* ei_trt_handle, int model_id, int warm_up_ms);

/**
 * @brief Copies input to the GPU (from CPU) for inference for model_id.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @param input a pointer to the (float) input
 * @param size the number of bytes to copy from the input
 * @return true if copying the input was successful.
 */
bool copyInputToDevice(EiTrt* ei_trt_handle, int model_id, float* input, int size);

/**
 * @brief Perform inference
 *
 * @param ei_trt_handle EI TensorRT context.
 * @return int 0 on success, <0 otherwise
 */
int infer(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Copies output to the CPU (from GPU) after inference from model_id.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @param output a pointer to the (float) output
 * @param size the amount of bytes to copy from the output
 * @return true if copying the output was successful.
 */
bool copyOutputToHost(EiTrt* ei_trt_handle, int model_id, float* output, int size);

/**
 * @brief Configures the maximum workspace that may be allocated
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param size workspace size in bytes.
 */
void setMaxWorkspaceSize(EiTrt *ei_trt_handle, int size);

/**
 * @brief Returns the current configured maximum workspace size.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @return the size of the workspace in bytes.
 */
int getMaxWorkspaceSize(EiTrt *ei_trt_handle);

/**
 * @brief Returns the input size (in features) of model_id.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @return the input size (in features).
 */
int getInputSize(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Returns the output size (in features) of model_id.
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @return the output size (in features).
 */
int getOutputSize(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Returns the latest inference latency in ms for model with id
 * (model_id) and context (ei_trt_handle).
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @return the inference time in ms.
 **/
uint64_t getInferenceMs(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Returns the latest inference latency in us for model with id
 * (model_id) and context (ei_trt_handle).
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @return the inference time in us.
 **/
uint64_t getInferenceUs(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Returns the latest inference latency in ns for model with id
 * (model_id) and context (ei_trt_handle).
 *
 * @param ei_trt_handle EI TensorRT context.
 * @param model_id a reference to the model to work on.
 * @return the inference time in ns.
 **/
uint64_t getInferenceNs(EiTrt* ei_trt_handle, int model_id);

/**
 * @brief Returns the current library major version
 *
 * @param ei_trt_handle EI TensorRT context.
 * @return the library's major version.
 **/
int getMajorVersion(EiTrt *ei_trt_handle);

/**
 * @brief Returns the current library minor version
 *
 * @param ei_trt_handle EI TensorRT context.
 * @return the library's minor version.
 **/
int getMinorVersion(EiTrt *ei_trt_handle);

/**
 * @brief Returns the current library patch version
 *
 * @param ei_trt_handle EI TensorRT context.
 * @return the library's patch version.
 **/
int getPatchVersion(EiTrt *ei_trt_handle);
}
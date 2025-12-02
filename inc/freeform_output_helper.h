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

#ifndef _FREEFORM_OUTPUT_HELPER_H_
#define _FREEFORM_OUTPUT_HELPER_H_

#include <vector>
#include <stdio.h>
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#if EI_CLASSIFIER_FREEFORM_OUTPUT

static std::map<ei_impulse_handle_t *, std::vector<matrix_t>> freeform_output_map;

/**
 * For "freeform" outputs, the application needs to allocate the memory (one matrix_t per output tensor).
 * Here we'll use a global map (one per impulse handle) to allocate this memory
 * (on the heap, using a matrix_t wrapper). In your own application you can just use something like this:
 *
 *     std::vector<matrix_t> freeform_outputs;
 *     freeform_outputs.reserve(ei_default_impulse.impulse->freeform_outputs_size);
 *     for (size_t ix = 0; ix < ei_default_impulse.impulse->freeform_outputs_size; ++ix) {
 *         freeform_outputs.emplace_back(ei_default_impulse.impulse->freeform_outputs[ix], 1);
 *     }
 *     EI_IMPULSE_ERROR set_freeform_res = ei_set_freeform_output(freeform_outputs.data(), freeform_outputs.size());
 *     if (set_freeform_res != EI_IMPULSE_OK) {
 *         printf("ei_set_freeform_output failed with %d\n", set_freeform_res);
 *         exit(1);
 *     }
 *
 * Where the memory is fully owned by your application (when freeform_outputs goes out of scope ->
 * memory is freed).
 */
void freeform_outputs_init(ei_impulse_handle_t *impulse_handle) {
    // make new object in the map based on handle (so this works with multiple impulses)
    auto& freeform_outputs = freeform_output_map[impulse_handle];

    // reserve memory using the matrix_t (allocs on the heap)
    freeform_outputs.reserve(impulse_handle->impulse->freeform_outputs_size);
    for (size_t ix = 0; ix < impulse_handle->impulse->freeform_outputs_size; ++ix) {
        freeform_outputs.emplace_back(impulse_handle->impulse->freeform_outputs[ix], 1);
    }
    // and set the freeform output
    EI_IMPULSE_ERROR set_freeform_res = ei_set_freeform_output(freeform_outputs.data(), freeform_outputs.size());
    if (set_freeform_res != EI_IMPULSE_OK) {
        printf("ei_set_freeform_output failed with %d\n", set_freeform_res);
        exit(1);
    }
}

#else

void freeform_outputs_init(ei_impulse_handle_t *impulse_handle) { }

#endif // EI_CLASSIFIER_FREEFORM_OUTPUT

#endif // _FREEFORM_OUTPUT_HELPER_H_

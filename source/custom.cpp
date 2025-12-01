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

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "inc/bitmap_helper.h"

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::string read_file(const char *filename) {
    FILE *f = (FILE*)fopen(filename, "r");
    if (!f) {
        printf("Cannot open file %s\n", filename);
        return "";
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    std::string ss;
    ss.resize(size);
    rewind(f);
    fread(&ss[0], 1, size, f);
    fclose(f);
    return ss;
}

// Draw bounding box overlays onto the original image (for anomaly detection / object detection)
static void create_debug_bmp(ei_impulse_result_t *result, std::vector<float> *raw_features_ptr) {
#if (EI_CLASSIFIER_OBJECT_DETECTION == 1) || (EI_CLASSIFIER_HAS_VISUAL_ANOMALY)
    std::vector<float> raw_features = *raw_features_ptr;

#if (EI_CLASSIFIER_OBJECT_DETECTION == 1)
    for (size_t ix = 0; ix < result->bounding_boxes_count; ix++) {
        auto bb = result->bounding_boxes[ix];
        if (bb.value == 0) {
            continue;
        }

        for (size_t x = bb.x; x < bb.x + bb.width; x++) {
            for (size_t y = bb.y; y < bb.y + bb.height; y++) {
                raw_features[(y * EI_CLASSIFIER_INPUT_WIDTH) + x] = (float)0x00ff00;
            }
        }
    }
#endif

#if (EI_CLASSIFIER_HAS_VISUAL_ANOMALY)
    for (size_t ix = 0; ix < result->visual_ad_count; ix++) {
        auto bb = result->visual_ad_grid_cells[ix];
        if (bb.value == 0) {
            continue;
        }

        for (size_t x = bb.x; x < bb.x + bb.width; x++) {
            for (size_t y = bb.y; y < bb.y + bb.height; y++) {
                raw_features[(y * EI_CLASSIFIER_INPUT_WIDTH) + x] = (float)0xff0000;
            }
        }
    }
#endif

    create_bitmap_file("debug.bmp", raw_features.data(), EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
#endif
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Requires one parameter (a comma-separated list of raw features, or a file pointing at raw features)\n");
        return 1;
    }

    std::string input = argv[1];
    if (!strchr(argv[1], ' ') && strchr(argv[1], '.')) { // looks like a filename
        input = read_file(argv[1]);
    }

    std::istringstream ss(input);
    std::string token;

    std::vector<float> raw_features;

    while (std::getline(ss, token, ',')) {
        raw_features.push_back(std::stof(trim(token)));
    }

    if (raw_features.size() != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printf("The size of your 'features' array is not correct. Expected %d items, but had %lu\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, raw_features.size());
        return 1;
    }

    run_classifier_init();

#if EI_CLASSIFIER_FREEFORM_OUTPUT
    // for "freeform" outputs, the application needs to allocate the memory (one matrix_t per output tensor)
    std::vector<matrix_t> freeform_outputs;
    freeform_outputs.reserve(ei_default_impulse.impulse->freeform_outputs_size);
    for (size_t ix = 0; ix < ei_default_impulse.impulse->freeform_outputs_size; ++ix) {
        freeform_outputs.emplace_back(ei_default_impulse.impulse->freeform_outputs[ix], 1);
    }
    EI_IMPULSE_ERROR set_freeform_res = ei_set_freeform_output(freeform_outputs.data(), freeform_outputs.size());
    if (set_freeform_res != EI_IMPULSE_OK) {
        printf("ei_set_freeform_output failed with %d\n", set_freeform_res);
        exit(1);
    }
#endif // EI_CLASSIFIER_FREEFORM_OUTPUT

    ei_impulse_result_t result;

    signal_t signal;
    numpy::signal_from_buffer(&raw_features[0], raw_features.size(), &signal);

    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
    if (res != EI_IMPULSE_OK) {
        printf("run_classifier failed (%d)\n", (int)res);
        return 1;
    }

    // Print results, see edge-impulse-sdk/classifier/ei_print_results.h
    ei_print_results(&ei_default_impulse, &result);

    create_debug_bmp(&result, &raw_features);
}

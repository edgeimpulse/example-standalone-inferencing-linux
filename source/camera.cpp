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

#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio_c.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "iostream"

static bool use_debug = false;

// If you don't want to allocate this much memory you can use a signal_t structure as well
// and read directly from a cv::Mat object, but on Linux this should be OK
static float features[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT];

/**
 * Resize and crop to the set width/height from model_metadata.h
 */
void resize_and_crop(cv::Mat *in_frame, cv::Mat *out_frame) {
    // to resize... we first need to know the factor
    float factor_w = static_cast<float>(EI_CLASSIFIER_INPUT_WIDTH) / static_cast<float>(in_frame->cols);
    float factor_h = static_cast<float>(EI_CLASSIFIER_INPUT_HEIGHT) / static_cast<float>(in_frame->rows);

    float largest_factor = factor_w > factor_h ? factor_w : factor_h;

    cv::Size resize_size(ceil(largest_factor * static_cast<float>(in_frame->cols)),
        ceil(largest_factor * static_cast<float>(in_frame->rows)));

    if (use_debug) {
        printf("resize_size width=%d height=%d\n", resize_size.width, resize_size.height);
    }

    cv::Mat resized;
    cv::resize(*in_frame, resized, resize_size);

    int crop_x = resize_size.width > resize_size.height ?
        (resize_size.width - resize_size.height) / 2 :
        0;
    int crop_y = resize_size.height > resize_size.width ?
        (resize_size.height - resize_size.width) / 2 :
        0;

    cv::Rect crop_region(crop_x, crop_y, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);

    if (use_debug) {
        printf("crop_region x=%d y=%d width=%d height=%d\n", crop_x, crop_y, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    }

    *out_frame = resized(crop_region);
}

int main(int argc, char** argv) {
    // If you see: OpenCV: not authorized to capture video (status 0), requesting... Abort trap: 6
    // This might be a permissions issue. Are you running this command from a simulated shell (like in Visual Studio Code)?
    // Try it from a real terminal.

    if (argc < 2) {
        printf("Requires one parameter (ID of the webcam).\n");
        printf("You can find these via `v4l2-ctl --list-devices`.\n");
        printf("E.g. for:\n");
        printf("    C922 Pro Stream Webcam (usb-70090000.xusb-2.1):\n");
        printf("            /dev/video0\n");
        printf("The ID of the webcam is 0\n");
        exit(1);
    }

    for (int ix = 2; ix < argc; ix++) {
        if (strcmp(argv[ix], "--debug") == 0) {
            printf("Enabling debug mode\n");
            use_debug = true;
        }
    }

    // open the webcam...
    cv::VideoCapture camera(atoi(argv[1]));
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // print camera properties
    printf("");
    printf("Camera properties:\n");
    printf("    width: %d\n", (int)camera.get(cv::CAP_PROP_FRAME_WIDTH));
    printf("    height: %d\n", (int)camera.get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("    fps: %d\n", (int)camera.get(cv::CAP_PROP_FPS));

    if (use_debug) {
        // create a window to display the images from the webcam
        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    }

    // this will contain the image from the webcam
    cv::Mat frame;
    run_classifier_init();

    // display the frame until you press a key
    while (1) {
        // 100ms. between inference
        int64_t next_frame = (int64_t)(ei_read_timer_ms() + 100);

        // capture the next frame from the webcam
        camera >> frame;

        cv::Mat cropped;
        resize_and_crop(&frame, &cropped);

        size_t feature_ix = 0;
        for (int rx = 0; rx < (int)cropped.rows; rx++) {
            for (int cx = 0; cx < (int)cropped.cols; cx++) {
                cv::Vec3b pixel = cropped.at<cv::Vec3b>(rx, cx);
                uint8_t b = pixel.val[0];
                uint8_t g = pixel.val[1];
                uint8_t r = pixel.val[2];
                features[feature_ix++] = (r << 16) + (g << 8) + b;
            }
        }

        ei_impulse_result_t result;

        // construct a signal from the features buffer
        signal_t signal;
        numpy::signal_from_buffer(features, EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT, &signal);

        // and run the classifier
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
        if (res != 0) {
            printf("ERR: Failed to run classifier (%d)\n", res);
            return 1;
        }

    // print the predictions
    printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);

    #if EI_CLASSIFIER_OBJECT_DETECTION == 1
        printf("#Object detection results:\n");
        bool found_bb = false;
        for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
            auto bb = result.bounding_boxes[ix];
            if (bb.value == 0) {
                continue;
            }

            found_bb = true;
            printf("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
        }

        if (!found_bb) {
            printf("    no objects found\n");
        }
    #else
        printf("#Classification results:\n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            printf("%s: %.05f\n", result.classification[ix].label, result.classification[ix].value);
        }
    #endif

    #if EI_CLASSIFIER_HAS_ANOMALY == 3 // visual AD
        printf("#Visual anomaly grid results:\n");
        for (uint32_t i = 0; i < result.visual_ad_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
            if (bb.value == 0) {
                continue;
            }
            printf("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
        }
        printf("Visual anomaly values: Mean %.3f Max %.3f\n", result.visual_ad_result.mean_value, result.visual_ad_result.max_value);
    #endif

        // print the open traces
        ei_printf("Open traces:\r\n");
        for (uint32_t i = 0; i < result.postprocessed_output.object_tracking_output.open_traces_count; i++) {
            ei_object_tracking_trace_t trace = result.postprocessed_output.object_tracking_output.open_traces[i];
            ei_printf("  Trace %d: %s [ x: %u, y: %u, width: %u, height: %u, value: %f ]\r\n",
                    trace.id,
                    trace.label,
                    trace.x,
                    trace.y,
                    trace.width,
                    trace.height,
                    trace.value);
        }

        // show the image on the window
        if (use_debug) {
            // draw the bounding boxes of the traces
            for (uint32_t i = 0; i < result.postprocessed_output.object_tracking_output.open_traces_count; i++) {
                ei_object_tracking_trace_t trace = result.postprocessed_output.object_tracking_output.open_traces[i];
                cv::rectangle(cropped, cv::Rect(trace.x, trace.y, trace.width, trace.height), cv::Scalar(0, 255, 0), 2);
                // add the label and ID
                cv::putText(cropped, std::to_string(trace.id) + ": " + trace.label, cv::Point(trace.x, trace.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("File", cropped);
            // wait (10ms) for a key to be pressed
            if (cv::waitKey(10) >= 0)
                break;
        }

        int64_t sleep_ms = next_frame > (int64_t)ei_read_timer_ms() ? next_frame - (int64_t)ei_read_timer_ms() : 0;
        if (sleep_ms > 0) {
            usleep(sleep_ms * 1000);
        }
    }
    run_classifier_deinit();
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor."
#endif

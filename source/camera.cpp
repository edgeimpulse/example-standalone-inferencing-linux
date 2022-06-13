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

#include <unistd.h>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio/videoio_c.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "iostream"

static bool use_debug = true;

// If you don't want to allocate this much memory you can use a signal_t structure as well
// and read directly from a cv::Mat object, but on Linux this should be OK
static float features[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT];

/**
 * Resize and crop to the set width/height from model_metadata.h
 */
void resize_and_crop(cv::Mat *in_frame, cv::Mat *out_frame, int width, int height) {
    // to resize... we first need to know the factor
    float factor_w = static_cast<float>(width) / static_cast<float>(in_frame->cols);
    float factor_h = static_cast<float>(height) / static_cast<float>(in_frame->rows);

    float largest_factor = factor_w > factor_h ? factor_w : factor_h;

    cv::Size resize_size(static_cast<int>(largest_factor * static_cast<float>(in_frame->cols)),
        static_cast<int>(largest_factor * static_cast<float>(in_frame->rows)));

    cv::Mat resized;
    cv::resize(*in_frame, resized, resize_size);

    int crop_x = resize_size.width > resize_size.height ?
        (resize_size.width - resize_size.height) / 2 :
        0;
    int crop_y = resize_size.height > resize_size.width ?
        (resize_size.height - resize_size.width) / 2 :
        0;

    cv::Rect crop_region(crop_x, crop_y, width, height);

    if (use_debug) {
        // printf("crop_region x=%d y=%d width=%d height=%d\n", crop_x, crop_y, width, height);
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

    if (use_debug) {
        // create a window to display the images from the webcam
        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    }

    // this will contain the image from the webcam
    cv::Mat frame;

    // display the frame until you press a key
    while (1) {
        // 100ms. between inference
        int64_t next_frame = (int64_t)(ei_read_timer_ms() + 100);

        // capture the next frame from the webcam
        camera >> frame;

        size_t oh = frame.rows < frame.cols ? frame.rows : frame.cols;

        cv::Mat cropped_original;
        resize_and_crop(&frame, &cropped_original, oh, oh);

        cv::Mat cropped;
        resize_and_crop(&frame, &cropped, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);

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

    #if EI_CLASSIFIER_OBJECT_DETECTION == 1
        // scaling factor between full-res video and output
        float bb_factor = static_cast<float>(cropped_original.rows) / static_cast<float>(cropped.rows);

        // this is BGR (not RGB)
        cv::Scalar colors[] = {
            cv::Scalar(75, 25, 230),
            cv::Scalar(75, 180, 60),
            cv::Scalar(25, 225, 255),
            cv::Scalar(216, 99, 67),
            cv::Scalar(49, 130, 245),
        };
        int max_colors = sizeof(colors) / sizeof(colors[0]);

        printf("Classification result (%d ms.):\n", result.timing.dsp + result.timing.classification);
        bool found_bb = false;
        for (size_t ix = 0; ix < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; ix++) {
            auto bb = result.bounding_boxes[ix];
            if (bb.value == 0) {
                continue;
            }

            int label_ix = 0;
            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                if (strcmp(bb.label, ei_classifier_inferencing_categories[ix]) == 0) {
                    label_ix = ((int)ix) % max_colors;
                    break;
                }
            }

            float bb_center_x = (static_cast<float>(bb.x) + (static_cast<float>(bb.width) * 0.5f)) * bb_factor;
            float bb_center_y = (static_cast<float>(bb.y) + (static_cast<float>(bb.height) * 0.5f)) * bb_factor;

            cv::Point bb_center(bb_center_x, bb_center_y);
            cv::circle(cropped_original, bb_center, 3.0f * bb_factor, colors[label_ix], cv::LINE_AA, 0.5f * bb_factor);

            double font_scale = 0.1f * bb_factor;
            int thickness = 0.25f * bb_factor;
            int baseline = 0;
            cv::Size text_size = getTextSize(bb.label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

            cv::Point text_coords(bb_center_x - (3.0f * bb_factor), bb_center_y - (4.5f * bb_factor));
            cv::Point text_top_left(text_coords.x - (1.0f * bb_factor), text_coords.y - text_size.height - (1.0f * bb_factor));
            cv::Point text_bottom_right(text_coords.x + text_size.width + (1.0f * bb_factor), text_coords.y + baseline + (1.0f * bb_factor));
            cv::rectangle(cropped_original, text_top_left, text_bottom_right, colors[label_ix], -1);

            cv::putText(cropped_original, bb.label, text_coords, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);

            found_bb = true;
            printf("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\n", bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
        }

        if (!found_bb) {
            printf("    no objects found\n");
        }
    #else
        printf("%d ms. ", result.timing.dsp + result.timing.classification);
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            printf("%s: %.05f", result.classification[ix].label, result.classification[ix].value);
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                printf(", ");
            }
        }
        printf("\n");
    #endif

        // show the image on the window
        if (use_debug) {
            cv::imshow("Webcam", cropped_original);
            // wait (10ms) for a key to be pressed
            if (cv::waitKey(10) >= 0)
                break;
        }

        int64_t sleep_ms = next_frame > (int64_t)ei_read_timer_ms() ? next_frame - (int64_t)ei_read_timer_ms() : 0;
        if (sleep_ms > 0) {
            usleep(sleep_ms * 1000);
        }
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor."
#endif

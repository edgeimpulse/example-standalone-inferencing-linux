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

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include <alsa/asoundlib.h>

// Forward declarations
int microphone_audio_signal_get_data(size_t, size_t, float *);

#define SLICE_LENGTH_MS      250        // 4 inferences per second
#define SLICE_LENGTH_VALUES  (EI_CLASSIFIER_RAW_SAMPLE_COUNT / (1000 / SLICE_LENGTH_MS))

static bool use_debug = false; // Set this to true to see e.g. features generated from the raw signal and log WAV files

static int16_t classifier_buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t)]; // full classifier buffer

// libalsa state
static snd_pcm_t *capture_handle;
static int channels = 1;
static unsigned int rate = EI_CLASSIFIER_FREQUENCY;
static snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;
static char *card;

/**
 * Initialize the alsa library
 */
int init_alsa(bool debug = false) {
    int err;

    snd_pcm_hw_params_t *hw_params;

    if ((err = snd_pcm_open(&capture_handle, card, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "cannot open audio device %s (%s)\n",
                card,
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "audio interface opened\n");
    }

    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params allocated\n");
    }

    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params)) < 0)
    {
        fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params initialized\n");
    }

    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        fprintf(stderr, "cannot set access type (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params access set\n");
    }

    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, format)) < 0)
    {
        fprintf(stderr, "cannot set format (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params format set\n");
    }

    if ((err = snd_pcm_hw_params_set_rate(capture_handle, hw_params, rate, 0)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n",
                snd_strerror(err));
        return 1;
    }
    else {
        unsigned int read_rate;
        int read_dir;

        snd_pcm_hw_params_get_rate(hw_params, &read_rate, &read_dir);

        if (debug) {
            fprintf(stdout, "hw_params rate set: %d\n", read_rate);
        }
    }

    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, channels)) < 0) {
        fprintf(stderr, "cannot set channel count (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params channels set:%d\n", channels);
    }

    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
        fprintf(stderr, "cannot set parameters (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "hw_params set\n");
    }

    snd_pcm_hw_params_free(hw_params);

    if (debug) {
        fprintf(stdout, "hw_params freed\n");
    }

    if ((err = snd_pcm_prepare(capture_handle)) < 0)
    {
        fprintf(stderr, "cannot prepare audio interface for use (%s)\n",
                snd_strerror(err));
        return 1;
    }

    if (debug) {
        fprintf(stdout, "audio interface prepared\n");
    }

    return 0;
}

void close_alsa(int signum) {
    snd_pcm_drop(capture_handle);
    snd_pcm_close(capture_handle);
    exit(0);
}

/**
 * Classify the current buffer
 */
void classify_current_buffer() {
    // write the WAV file for debug purposes...
    if (use_debug) {
        char filename[128] = { 0 };

        static int classify_counter = 0;
        struct stat st = { 0 };
        if (stat("out", &st) == -1) {
            mkdir("out", 0700);
        }

        uint32_t wavFreq = EI_CLASSIFIER_FREQUENCY;
        uint32_t fileSize = 44 + (EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
        uint32_t dataSize = (EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
        uint32_t srBpsC8 = (wavFreq * 16 * 1) / 8;

        uint8_t wav_header[44] = {
            0x52, 0x49, 0x46, 0x46, // RIFF
            (uint8_t)(fileSize & 0xff), (uint8_t)((fileSize >> 8) & 0xff), (uint8_t)((fileSize >> 16) & 0xff), (uint8_t)((fileSize >> 24) & 0xff),
            0x57, 0x41, 0x56, 0x45, // WAVE
            0x66, 0x6d, 0x74, 0x20, // fmt
            0x10, 0x00, 0x00, 0x00, // length of format data
            0x01, 0x00, // type of format (1=PCM)
            0x01, 0x00, // number of channels
            (uint8_t)(wavFreq & 0xff), (uint8_t)((wavFreq >> 8) & 0xff), (uint8_t)((wavFreq >> 16) & 0xff), (uint8_t)((wavFreq >> 24) & 0xff),
            (uint8_t)(srBpsC8 & 0xff), (uint8_t)((srBpsC8 >> 8) & 0xff), (uint8_t)((srBpsC8 >> 16) & 0xff), (uint8_t)((srBpsC8 >> 24) & 0xff),
            0x02, 0x00, 0x10, 0x00,
            0x64, 0x61, 0x74, 0x61, // data
            (uint8_t)(dataSize & 0xff), (uint8_t)((dataSize >> 8) & 0xff), (uint8_t)((dataSize >> 16) & 0xff), (uint8_t)((dataSize >> 24) & 0xff),
        };

        snprintf(filename, 128, "out/data.%d.wav", ++classify_counter);

        FILE *f = fopen(filename, "w+");
        if (!f) {
            printf("Failed to create file '%s'\n", filename);
            return;
        }
        fwrite(wav_header, 1, 44, f);
        fwrite(classifier_buffer, 2, EI_CLASSIFIER_RAW_SAMPLE_COUNT, f);
        fclose(f);
    }

    // classify the current buffer and print the results
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, use_debug);
    if (r != EI_IMPULSE_OK) {
        printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    printf("%d ms. ", result.timing.dsp + result.timing.classification);
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        printf("%s: %.05f", result.classification[ix].label, result.classification[ix].value);
        if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

/**
 * @brief      main function. Runs the inferencing loop.
 */
int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Requires one parameter (ID of the sound card) in the form of plughw:1,0 (where 1=card number, 0=device).\n");
        printf("You can find these via `cat /proc/asound/cards`. E.g. for:\n");
        printf("   0 [Headphones     ]: bcm2835_headphonbcm2835 Headphones - bcm2835 Headphones\n");
        printf("                        bcm2835 Headphones\n");
        printf("   1 [Webcam         ]: USB-Audio - C922 Pro Stream Webcam\n");
        printf("                        C922 Pro Stream Webcam at usb-0000:01:00.0-1.3, high speed\n");
        printf("The ID for 'C922 Pro Stream Webcam' is then plughw:1,0\n");
        exit(1);
    }

    card = argv[1];

    for (int ix = 2; ix < argc; ix++) {
        if (strcmp(argv[ix], "--debug") == 0) {
            printf("Enabling debug mode\n");
            use_debug = true;
        }
    }


    if (init_alsa(use_debug) != 0) {
        exit(1);
    }

    signal(SIGINT, close_alsa);

    // allocate buffers for the slice
    int16_t slice_buffer[SLICE_LENGTH_VALUES * sizeof(int16_t)];
    uint32_t classify_count = 0;

    while (1) {
        int x = snd_pcm_readi(capture_handle, slice_buffer, SLICE_LENGTH_VALUES);
        if (x != SLICE_LENGTH_VALUES) {
            printf("Failed to read audio data (%d)\n", x);
            return 1;
        }

        // so let's say we have a 16000 element classifier_buffer
        // then we want to move 4000..16000 to position 0..12000
        // and fill 12000..16000 with the data from slice_buffer

        // 1. roll -SLICE_LENGTH_VALUES here
        numpy::roll(classifier_buffer, EI_CLASSIFIER_RAW_SAMPLE_COUNT, -SLICE_LENGTH_VALUES);

        // 2. copy slice buffer to the end
        const size_t classifier_buffer_offset = EI_CLASSIFIER_RAW_SAMPLE_COUNT - SLICE_LENGTH_VALUES;
        memcpy(classifier_buffer + classifier_buffer_offset, slice_buffer, SLICE_LENGTH_VALUES * sizeof(int16_t));

        // ignore the first N slices we classify, we don't have a complete frame yet
        if (++classify_count < EI_CLASSIFIER_RAW_SAMPLE_COUNT / SLICE_LENGTH_VALUES) {
            continue;
        }

        // 3. and classify!
        classify_current_buffer();
    }

    close_alsa(0);
}

/**
 * Get data from the classifier buffer
 */
int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    return numpy::int16_to_float(classifier_buffer + offset, out_ptr, length);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

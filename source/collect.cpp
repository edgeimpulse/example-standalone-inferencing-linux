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

#include <time.h>
#include "inc/httplib.h"
#include "ingestion-sdk-c/inc/sensor_aq.h"
#include "ingestion-sdk-c/inc/signing/sensor_aq_mbedtls_hs256.h"

// Your credentials here, you can find these via **Dashboard > Keys** in your Edge Impulse project
const char *API_KEY = "ei_...";
const char *HMAC_KEY = "...";

int main() {

    // The sensor format supports signing the data, set up a signing context
    sensor_aq_signing_ctx_t signing_ctx;

    // We'll use HMAC SHA256 signatures, which can be created through Mbed TLS
    // If you use a different crypto library you can implement your own context
    sensor_aq_mbedtls_hs256_ctx_t hs_ctx;

    // Set up the context, the last argument is the HMAC key
    sensor_aq_init_mbedtls_hs256_context(&signing_ctx, &hs_ctx, HMAC_KEY);

    // Set up the sensor acquisition basic context
    sensor_aq_ctx ctx = {
        // We need a single buffer. The library does not require any dynamic allocation (but your TLS library might)
        { (unsigned char*)malloc(1024), 1024 },

        // Pass in the signing context
        &signing_ctx,

        // And pointers to fwrite and fseek - note that these are pluggable so you can work with them on
        // non-POSIX systems too. Just override the EI_SENSOR_AQ_STREAM macro to your custom file type.
        &fwrite,
        &fseek,
        // if you set the time function this will add 'iat' (issued at) field to the header with the current time
        // if you don't include it, this will be omitted
        &time
    };

    // Payload header
    sensor_aq_payload_info payload = {
        // Unique device ID (optional), set this to e.g. MAC address or device EUI **if** your device has one, otherwise you can leave it empty
        "00:00:00:00:00:00",
        // Device type (required), use the same device type for similar devices
        "LINUX_TEST",
        // How often new data is sampled in ms. (100Hz = every 10 ms.)
        10,
        // The axes which you'll use (name, units)
        { { "accX", "m/s2" }, { "accY", "m/s2" }, { "accZ", "m/s2" } }
    };

    // Place to write our data.
    // The library streams data, and does not cache everything in buffers
    FILE *file = tmpfile();

    // Initialize the context, this verifies that all requirements are present
    // it also writes the initial CBOR structure
    int res;
    res = sensor_aq_init(&ctx, &payload, file, false);
    if (res != AQ_OK) {
        printf("sensor_aq_init failed (%d)\n", res);
        return 1;
    }

    // Periodically call `sensor_aq_add_data` (every 10 ms. in this example) to append data
    for (int ix = 0; ix < 100 * 2; ix++) {
        float values[3] = {
            (float) sin((float)ix * 0.1f) * 10.0f,
            (float) cos((float)ix * 0.1f) * 10.0f,
            (float) (sin((float)ix * 0.1f) + cos((float)ix * 0.1f)) * 10.0f,
        };

        res = sensor_aq_add_data(&ctx, values, 3);
        if (res != AQ_OK) {
            printf("sensor_aq_add_data failed (%d)\n", res);
            return 1;
        }
    }

    // When you're done call sensor_aq_finish - this will calculate the finalized signature and close the CBOR file
    res = sensor_aq_finish(&ctx);
    if (res != AQ_OK) {
        printf("sensor_aq_finish failed (%d)\n", res);
        return 1;
    }

    // Print the content of the file here:
    fseek(file, 0, SEEK_END);
    size_t len = ftell(file);
    uint8_t *buffer = (uint8_t*)malloc(len);

    fseek(file, 0, SEEK_SET);
    fread(buffer, len, 1, file);

    // For convenience we'll write the encoded file. You can throw this directly in http://cbor.me to decode (uncomment the next 5 lines for that)
    // printf("Encoded file:\n");
    // for (size_t ix = 0; ix < len; ix++) {
    //     printf("%02x ", buffer[ix]);
    // }
    // printf("\n");

    // Upload the file...
    httplib::Client cli("http://ingestion.edgeimpulse.com");
    httplib::Headers headers = {
        { "x-api-key", API_KEY },
        { "x-file-name", "linuxtest01.cbor" },
        { "x-label", "linuxtest" },
        { "x-disallow-duplicates", "1" },
        { "content-type", "application/cbor" }
    };
    // you can replace 'training' here with 'testing'
    auto http_res = cli.Post("/api/training/data", headers, (const char*)buffer, len, "application/cbor");
    printf("Uploaded data, status=%d, body=%s\n", http_res->status, http_res->body.c_str());
}

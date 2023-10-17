#include <stdio.h>
#include <stdarg.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "json/json.hpp"
#include "rapidjson/document.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace std;

#define STDIN_BUFFER_SIZE       (10 * 1024 * 1024)
#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_AKIDA)
#include "pybind11/embed.h"
namespace py = pybind11;
extern std::stringstream engine_info;
#elif ((EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_MEMRYX) && \
      (EI_CLASSIFIER_USE_MEMRYX_SOFTWARE == 1))
#include "pybind11/embed.h"
using namespace pybind11::literals; // to bring in the `_a` literal
namespace py = pybind11;
extern std::stringstream engine_info;
#elif ((EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_MEMRYX) && \
      (EI_CLASSIFIER_USE_MEMRYX_HARDWARE == 1))
extern std::stringstream engine_info;
#else
std::stringstream engine_info;
#endif

typedef struct {
    bool initialized;
    int version;
} runner_state_t;

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

static char rapidjson_buffer[10 * 1024 * 1024] ALIGN(8);
rapidjson::MemoryPoolAllocator<> rapidjson_allocator(rapidjson_buffer, sizeof(rapidjson_buffer));

static runner_state_t state = { 0 };

void json_send_classification_response(int id,
                                       uint64_t json_parsing_ms,
                                       uint64_t stdin_ms,
                                       EI_IMPULSE_ERROR res,
                                       ei_impulse_result_t *result_ptr,
                                       char *resp_buffer,
                                       size_t resp_buffer_size)
{
    ei_impulse_result_t result = *result_ptr;

    if (res != 0) {
        char err_msg[128];
        snprintf(err_msg, 128, "Classifying failed, error code was %d", (int)res);

        nlohmann::json err = {
            {"id", id},
            {"success", false},
            {"error", err_msg},
        };
        snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
        return;
    }

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    nlohmann::json bb_res = nlohmann::json::array();
    for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
        auto bb = result.bounding_boxes[ix];
        if (bb.value == 0) {
            continue;
        }
        nlohmann::json bb_json = {
            {"label", bb.label},
            {"value", bb.value},
            {"x", bb.x},
            {"y", bb.y},
            {"width", bb.width},
            {"height", bb.height},
        };
        bb_res.push_back(bb_json);
    }
#else
    nlohmann::json classify_res;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        classify_res[result.classification[ix].label] = result.classification[ix].value;
    }
#endif

    nlohmann::json resp = {
        {"id", id},
        {"success", true},
        {"result", {
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
            {"bounding_boxes", bb_res},
#else
            {"classification", classify_res},
#endif // EI_CLASSIFIER_OBJECT_DETECTION
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            {"anomaly", result.anomaly},
#endif // EI_CLASSIFIER_HAS_ANOMALY == 1
        }},
        {"timing", {
            {"dsp", result.timing.dsp},
            {"classification", result.timing.classification},
            {"anomaly", result.timing.anomaly},
            {"json", json_parsing_ms},
            {"stdin", stdin_ms},
        }},
    };
    if (engine_info.str().length() > 0) {
        resp["info"] = engine_info.str();
    }
    snprintf(resp_buffer, resp_buffer_size, "%s\n", resp.dump().c_str());
}

void json_message_handler(rapidjson::Document &msg, char *resp_buffer, size_t resp_buffer_size, uint64_t json_parsing_ms, uint64_t stdin_ms) {
    rapidjson::Value& id_v = msg["id"];
    if (!id_v.IsInt()) {
        nlohmann::json err = {
            {"success", false},
            {"error", "Missing 'id' field in message"},
        };
        snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
        return;
    }

    auto id = id_v.GetInt();

    rapidjson::Value& version = msg["hello"];
    rapidjson::Value& classify_data = msg["classify"];
    rapidjson::Value& classify_data_continuous = msg["classify_continuous"];

    if (version.IsInt()) {
        if (state.initialized) {
            nlohmann::json err = {
                {"id", id},
                {"success", false},
                {"error", "Invalid message 'hello', already initialized"},
            };
            snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
            return;
        }

        if (version.GetInt() != 1) {
            nlohmann::json err = {
                {"id", id},
                {"success", false},
                {"error", "Invalid 'version', only 1 supported"},
            };
            snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
            return;
        }

        run_classifier_init();

        vector<std::string> labels;
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            labels.push_back(std::string(ei_classifier_inferencing_categories[ix]));
        }

        // 1 image DSP block?
        int16_t channel_count = 0;
        if (ei_dsp_blocks_size == 1 && ei_dsp_blocks[0].extract_fn == &extract_image_features) {
            ei_dsp_config_image_t *config = (ei_dsp_config_image_t *)(ei_dsp_blocks[0].config);
            channel_count = strcmp(config->channels, "Grayscale") == 0 ? 1 : 3;
        }
        // other DSP block but image input? always assume 3 channels, we can't take shortcut here
        // anyway
        else if (EI_CLASSIFIER_INPUT_WIDTH != 0) {
            channel_count = 3;
        }

#if EI_CLASSIFIER_OBJECT_DETECTION
    #if EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER == EI_CLASSIFIER_LAST_LAYER_FOMO
        const char *model_type = "constrained_object_detection";
    #elif EI_CLASSIFIER_OBJECT_DETECTION
        const char *model_type = "object_detection";
    #endif // EI_CLASSIFIER_OBJECT_DETECTION_LAST_LAYER
#else
    const char *model_type = "classification";
#endif // EI_CLASSIFIER_OBJECT_DETECTION

        nlohmann::json resp = {
            {"id", id},
            {"success", true},
            {"project", {
                {"id", EI_CLASSIFIER_PROJECT_ID},
                {"owner", std::string(EI_CLASSIFIER_PROJECT_OWNER)},
                {"name", std::string(EI_CLASSIFIER_PROJECT_NAME)},
                {"deploy_version", EI_CLASSIFIER_PROJECT_DEPLOY_VERSION},
            }},
            {"model_parameters", {
                {"input_features_count", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE},
                {"sensor", EI_CLASSIFIER_SENSOR},
                {"frequency", EI_CLASSIFIER_FREQUENCY},
                {"interval_ms", EI_CLASSIFIER_INTERVAL_MS},
                {"axis_count", EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME},
                {"image_input_width", EI_CLASSIFIER_INPUT_WIDTH},
                {"image_input_height", EI_CLASSIFIER_INPUT_HEIGHT},
                {"image_input_frames", EI_CLASSIFIER_INPUT_FRAMES},
                {"image_channel_count", channel_count},
                {"label_count", EI_CLASSIFIER_LABEL_COUNT},
                {"has_anomaly", EI_CLASSIFIER_HAS_ANOMALY},
                {"labels", labels},
                {"model_type", model_type},
                {"slice_size", EI_CLASSIFIER_SLICE_SIZE},
                {"use_continuous_mode", EI_CLASSIFIER_SENSOR == EI_CLASSIFIER_SENSOR_MICROPHONE},
            }},
        };

        snprintf(resp_buffer, resp_buffer_size, "%s\n", resp.dump().c_str());

        state.initialized = true;
        state.version = version.GetInt();
    }
    else if (!state.initialized) {
        nlohmann::json err = {
            {"id", id},
            {"success", false},
            {"error", "Invalid message, should initialize first"},
        };
        snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
        return;
    }
    else if (classify_data.IsArray()) {
        vector<float> input_features;

        for (rapidjson::SizeType i = 0; i < classify_data.Size(); i++) {
            if (!classify_data[i].IsNumber()) {
                nlohmann::json err = {
                    {"id", id},
                    {"success", false},
                    {"error", "Failed to parse classify array, should contain all numbers"},
                };
                snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
                return;
            }
            input_features.push_back((float)classify_data[i].GetDouble());
        }

        if (input_features.size() != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
            char err_msg[128];
            snprintf(err_msg, 128, "Invalid number of features in 'classify', expected %d but got %d",
                (int)EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, (int)input_features.size());

            nlohmann::json err = {
                {"id", id},
                {"success", false},
                {"error", err_msg},
            };
            snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
            return;
        }

        ei_impulse_result_t result;
        memset(&result, 0, sizeof(ei_impulse_result_t));
        signal_t signal;
        numpy::signal_from_buffer(&input_features[0], input_features.size(), &signal);

        bool debug = false;
        rapidjson::Value &debug_v = msg["debug"];
        if (debug_v.IsBool()) {
            debug = debug_v.GetBool();
        }

        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, debug);
        json_send_classification_response(id, json_parsing_ms, stdin_ms,
            res, &result, resp_buffer, resp_buffer_size);
    }
    else if (classify_data_continuous.IsArray()) {
        vector<float> input_features;

        for (rapidjson::SizeType i = 0; i < classify_data_continuous.Size(); i++) {
            if (!classify_data_continuous[i].IsNumber()) {
                nlohmann::json err = {
                    {"id", id},
                    {"success", false},
                    {"error", "Failed to parse classify array, should contain all numbers"},
                };
                snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
                return;
            }
            input_features.push_back((float)classify_data_continuous[i].GetDouble());
        }

        if (input_features.size() != EI_CLASSIFIER_SLICE_SIZE) {
            char err_msg[128];
            snprintf(err_msg, 128, "Invalid number of features in 'classify_continuous', expected %d but got %d",
                (int)EI_CLASSIFIER_SLICE_SIZE, (int)input_features.size());

            nlohmann::json err = {
                {"id", id},
                {"success", false},
                {"error", err_msg},
            };
            snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
            return;
        }

        ei_impulse_result_t result;
        memset(&result, 0, sizeof(ei_impulse_result_t));
        signal_t signal;
        numpy::signal_from_buffer(&input_features[0], input_features.size(), &signal);

        bool debug = false;
        rapidjson::Value &debug_v = msg["debug"];
        if (debug_v.IsBool()) {
            debug = debug_v.GetBool();
        }

        EI_IMPULSE_ERROR res = run_classifier_continuous(&signal, &result, debug, true);
        json_send_classification_response(id, json_parsing_ms, stdin_ms,
            res, &result, resp_buffer, resp_buffer_size);
    }
    else {
        nlohmann::json err = {
            {"id", id},
            {"success", false},
            {"error", "Failed to handle message"},
        };
        snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
        return;
    }
}

int stdin_main() {
    static char *stdin_buffer = (char *)malloc(STDIN_BUFFER_SIZE);
    static char *response_buffer = (char *)calloc(STDIN_BUFFER_SIZE, 1);
    if (!stdin_buffer || !response_buffer) {
        printf("ERR: Could not allocate stdin_buffer or response_buffer\n");
        return 1;
    }
    static size_t stdin_buffer_ix = 0;
    static size_t open_count = 0;
    static size_t close_count = 0;
    uint64_t read_from_stdin_start = 0;

    char c;

    while ((c = getchar()) && c != EOF) {
        stdin_buffer[stdin_buffer_ix++] = c;

        if (stdin_buffer_ix > STDIN_BUFFER_SIZE - 1) {
            printf("Invalid message, received more than %d bytes, and no valid JSON message detected\n",
                STDIN_BUFFER_SIZE);
            return 1;
        }

        if (c == '{') {
            open_count++;
        }
        else if (c == '}') {
            close_count++;
            if (close_count == open_count) {
                uint64_t read_from_stdin = ei_read_timer_ms() - read_from_stdin_start;
                try {
                    auto now = ei_read_timer_ms();

                    rapidjson::Document msg(&rapidjson_allocator);
                    msg.Parse(stdin_buffer);

                    // auto msg = json::parse(stdin_buffer);
                    auto json_parsing_ms = ei_read_timer_ms() - now;
                    json_message_handler(msg, response_buffer, STDIN_BUFFER_SIZE, json_parsing_ms, read_from_stdin);
                    printf("%s", response_buffer);

                    rapidjson_allocator.Clear();
                }
                catch (const std::exception& e) {
                    nlohmann::json err = {
                        {"error", e.what()},
                    };
                    printf("%s\n", err.dump().c_str());
                }

                stdin_buffer_ix = 0;
                memset(stdin_buffer, 0, STDIN_BUFFER_SIZE);
                close_count = 0;
                open_count = 0;
            }
        }
        else if (open_count == 0) {
            stdin_buffer_ix--;
            read_from_stdin_start = 0;
        }

        if (open_count == 1 && read_from_stdin_start == 0) {
            read_from_stdin_start = ei_read_timer_ms();
        }
    }

    return 0;
}

int socket_main(char *socket_path) {
    int fd = socket(PF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        printf("ERR: Failed to create a new UNIX socket\n");
        return 1;
    }

    struct sockaddr_un addr = { 0 };
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socket_path);
    unlink(socket_path);
    int ret = ::bind(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (ret < 0) {
        printf("ERR: Failed to bind UNIX socket\n");
        return 1;
    }

	ret = ::listen(fd, 10);
    if (ret < 0) {
        printf("ERR: Failed to listen on UNIX socket\n");
        return 1;
    }

    // @todo: should we put this in a while(1) and move the accept() to another thread so you can have multiple listeners?
    // probably not
    printf("Waiting for connection on %s...\n", socket_path);
    int connfd = accept(fd, (struct sockaddr*)NULL, NULL);
    printf("Connected\n");

    char *socket_buffer = (char *)malloc(STDIN_BUFFER_SIZE);
    char *stdin_buffer = (char *)malloc(STDIN_BUFFER_SIZE);
    char *response_buffer = (char *)malloc(STDIN_BUFFER_SIZE);
    if (!socket_buffer || !stdin_buffer || !response_buffer) {
        printf("ERR: Could not allocate buffers\n");
        return 1;
    }
    static size_t stdin_buffer_ix = 0;
    static size_t open_count = 0;
    static size_t close_count = 0;
    uint64_t read_from_stdin_start = 0;

    int len;
    while ((len = read(connfd, socket_buffer, STDIN_BUFFER_SIZE)) > 0) {
        for (int ix = 0; ix < len; ix++) {
            char c = socket_buffer[ix];

            stdin_buffer[stdin_buffer_ix++] = c;

            if (stdin_buffer_ix > STDIN_BUFFER_SIZE - 1) {
                printf("Invalid message, received more than %d bytes, and no valid JSON message detected\n",
                    STDIN_BUFFER_SIZE);
                return 1;
            }

            if (c == '{') {
                open_count++;
            }
            else if (c == '}') {
                close_count++;
                if (close_count == open_count) {
                    uint64_t read_from_stdin = ei_read_timer_ms() - read_from_stdin_start;
                    try {
                        // printf("Incoming message: %s\n", stdin_buffer);
                        auto now = ei_read_timer_ms();
                        rapidjson::Document msg(&rapidjson_allocator);
                        msg.Parse(stdin_buffer);
                        // auto msg = json::parse(stdin_buffer);
                        auto json_parsing_ms = ei_read_timer_ms() - now;
                        json_message_handler(msg, response_buffer, STDIN_BUFFER_SIZE, json_parsing_ms, read_from_stdin);
                        // printf("Sending back: %s\n", response_buffer);
                        int ret = write(connfd, response_buffer, strlen(response_buffer) + 1);
                        if (ret < 0) {
                            printf("ERR: Failed to send message back (%d)\n", ret);
                        }
                        rapidjson_allocator.Clear();
                    }
                    catch (const std::exception& e) {
                        nlohmann::json err = {
                            {"error", e.what()},
                        };
                        snprintf(response_buffer, STDIN_BUFFER_SIZE, "%s\n", err.dump().c_str());

                        // printf("Sending back: %s\n", response_buffer);

                        int ret = write(connfd, response_buffer, strlen(response_buffer) + 1);
                        if (ret < 0) {
                            printf("ERR: Failed to send message back (%d)\n", ret);
                        }
                    }

                    stdin_buffer_ix = 0;
                    memset(stdin_buffer, 0, STDIN_BUFFER_SIZE);
                    close_count = 0;
                    open_count = 0;
                }
            }
            else if (open_count == 0) {
                stdin_buffer_ix--;
                read_from_stdin_start = 0;
            }

            if (open_count == 1 && read_from_stdin_start == 0) {
                read_from_stdin_start = ei_read_timer_ms();
            }
        }
    }

    close(connfd);

    return close(fd);
}

string trim(const string& str) {
    size_t first = str.find_first_not_of(' ');
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 2) {
        printf("Requires one parameter (either: 'stdin' or the name of a socket)\n");
        return 1;
    }

    state.initialized = false;

    if (strcmp(argv[1], "stdin") == 0) {
        printf("Edge Impulse Linux impulse runner - listening for JSON messages on stdin\n");
        return stdin_main();
    }
#if (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_AKIDA)
    else if (strcmp(argv[1], "debug") == 0) {
        py::scoped_interpreter guard{};
        py::module_ sys = py::module_::import("sys");
        ei_printf("DEBUG: sys.path:");
        for (py::handle p: sys.attr("path")) {
            ei_printf("\t%s\n", p.cast<std::string>().c_str());
        }
        return 0;
    }
#elif (EI_CLASSIFIER_INFERENCING_ENGINE == EI_CLASSIFIER_MEMRYX)
    else if (strcmp(argv[1], "debug") == 0) {
        printf("memryx inferencing engine selected\n");
        return 0;
    }
#endif
    else {
        printf("Edge Impulse Linux impulse runner - listening for JSON messages on socket '%s'\n", argv[1]);
        return socket_main(argv[1]);
    }
}


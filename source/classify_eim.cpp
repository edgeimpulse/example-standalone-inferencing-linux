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
#include <sys/socket.h>
#include <sys/un.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>
#include "json/json.hpp"
#include "rapidjson/document.h"

class EimRunner {
public:
    EimRunner(std::string model_file): _model_file(model_file), _socket_file("/tmp/eim.sock.XXXXXX"), _model_thread(NULL) {
        this->_initialized = false;
        this->_socket_fd = -1;
    }

    ~EimRunner() {
        if (this->_socket_fd != -1) {
            close(this->_socket_fd);
        }
        if (this->_model_thread) {
            // There's probably a much better way of doing this, but this works at least to kill the underlying .eim file
            char kill_cmd[1024];
            snprintf(kill_cmd, 1024, "ps aux | grep %s | awk '{print $2}' | sort | head -n 1 | xargs kill -9", this->_socket_file);
            system(kill_cmd);

            delete _model_thread;
        }
    }

    int hello(char *recv_buffer, size_t recv_buffer_size) {
        if (this->_initialized) {
            printf("ERR: EimRunner is already initialized\n");
            return -1;
        }

        int fd = mkstemp(this->_socket_file);
        if (fd == -1) {
            printf("ERR: Failed to open temporary file\n");
            return -1;
        }

        // close the file descriptor
        close(fd);

        _model_thread = new std::thread(&EimRunner::model_thread_main, this);

        for (int ix = 0; ix < 10; ix++) {
            sleep(1);
            printf("Checking if socket exists? %d\n", socket_file_exists());
            if (socket_file_exists()) {
                break;
            }
        }

        if (!socket_file_exists()) {
            printf("ERR: Socket file does not exist\n");
            return -1;
        }

        fd = socket(PF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            printf("ERR: Failed to create a new UNIX socket\n");
            return -1;
        }

        struct sockaddr_un addr = { 0 };
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, this->_socket_file);

        int ret = ::connect(fd, (struct sockaddr *)&addr, sizeof(addr));
        if (ret < 0) {
            printf("ERR: Failed to connect to UNIX socket (%d)\n", ret);
            return 1;
        }

        // printf("Connected! (%d)\n", ret);

        nlohmann::json hello = {
            {"id", 1},
            {"hello", 1},
        };
        ret = ::send(fd, hello.dump().c_str(), strlen(hello.dump().c_str()) + 1, 0);
        if (ret < 0) {
            printf("ERR: Failed to send message to UNIX socket (%d)\n", ret);
            return 1;
        }

        ret = ::recv(fd, recv_buffer, recv_buffer_size, 0);
        if (ret < 0) {
            printf("ERR: Failed to receive message (%d)\n", ret);
            return 1;
        }

        this->_socket_fd = fd;

        return 0;
    }

    int classify(std::vector<float> features, char *recv_buffer, size_t recv_buffer_size) {
        nlohmann::json hello = {
            {"id", 2},
            {"classify", features},
        };
        int ret = ::send(this->_socket_fd, hello.dump().c_str(), strlen(hello.dump().c_str()) + 1, 0);
        if (ret < 0) {
            printf("ERR: Failed to send message to UNIX socket (%d)\n", ret);
            return 1;
        }

        ret = ::recv(this->_socket_fd, recv_buffer, recv_buffer_size, 0);
        if (ret < 0) {
            printf("ERR: Failed to receive message (%d)\n", ret);
            return 1;
        }

        return 0;
    }

private:
    void model_thread_main() {
        char command[2048];
        snprintf(command, 2048, "%s %s > /dev/null", this->_model_file.c_str(), this->_socket_file);

        // printf("Command is: %s\n", command);

        system(command);
    }

    bool socket_file_exists() {
        return (access(this->_socket_file, F_OK) != -1);
    }

    std::string _model_file;
    char _socket_file[64];
    std::thread *_model_thread;
    bool _initialized;
    int _socket_fd;
};

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

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Requires two parameter (an .eim file, and a features file)\n");
        return 1;
    }

    std::string model_file = argv[1];
    std::string input = read_file(argv[2]);

    std::istringstream ss(input);
    std::string token;

    std::vector<float> raw_features;

    while (std::getline(ss, token, ',')) {
        raw_features.push_back(std::stof(trim(token)));
    }

    char recv_buffer[8192];

    EimRunner runner(model_file);
    int res = runner.hello(recv_buffer, sizeof(recv_buffer));
    printf("hello_res = %d, %s\n", res, recv_buffer);

    res = runner.classify(raw_features, recv_buffer, sizeof(recv_buffer));
    printf("classify_res = %d, %s\n", res, recv_buffer);
}

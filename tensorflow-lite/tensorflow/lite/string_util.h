/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Util methods to read and write String tensors.
// String tensors are considered to be char tensor with protocol.
//   [0, 3] 4 bytes: N, num of strings in the tensor in little endian.
//   [(i+1)*4, (i+1)*4+3] 4 bytes: offset of i-th string in little endian,
//                                 for i from 0 to N-1.
//   [(N+1)*4, (N+1)*4+3] 4 bytes: length of the whole char buffer.
//   [offset(i), offset(i+1) - 1] : content of i-th string.
// Example of a string tensor:
// [
//   2, 0, 0, 0,     # 2 strings.
//   16, 0, 0, 0,    # 0-th string starts from index 16.
//   18, 0, 0, 0,    # 1-st string starts from index 18.
//   18, 0, 0, 0,    # total length of array.
//   'A', 'B',       # 0-th string [16..17]: "AB"
// ]                 # 1-th string, empty
//
// A typical usage:
// In op.Eval(context, node):
//   DynamicBuffer buf;
//   # Add string "AB" to tensor, string is stored in dynamic buffer.
//   buf.AddString("AB", 2);
//   # Write content of DynamicBuffer to tensor in format of string tensor
//   # described above.
//   buf.WriteToTensor(tensor, nullptr)

#ifndef TENSORFLOW_LITE_STRING_UTIL_H_
#define TENSORFLOW_LITE_STRING_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <vector>

#include "tensorflow-lite/tensorflow/lite/core/c/common.h"
#include "tensorflow-lite/tensorflow/lite/string_type.h"

namespace tflite {

// Convenient structure to store string pointer and length. Note that
// methods on DynamicBuffer enforce that the whole buffer (and by extension
// every contained string) is of max length (2ul << 30) - 1. See
// string_util.cc for more info.
typedef struct {
  const char* str;
  size_t len;
} StringRef;

constexpr uint64_t kDefaultMaxLength = std::numeric_limits<int>::max();

// DynamicBuffer holds temporary buffer that will be used to create a dynamic
// tensor. A typical usage is to initialize a DynamicBuffer object, fill in
// content and call CreateStringTensor in op.Eval().
class DynamicBuffer {
 public:
  explicit DynamicBuffer(size_t max_length = kDefaultMaxLength)
      : offset_({0}), max_length_(max_length) {}

  // Add string to dynamic buffer by resizing the buffer and copying the data.
  TfLiteStatus AddString(const StringRef& string);

  // Add string to dynamic buffer by resizing the buffer and copying the data.
  TfLiteStatus AddString(const char* str, size_t len);

  // Join a list of string with separator, and add as a single string to the
  // buffer.
  void AddJoinedString(const std::vector<StringRef>& strings, char separator);
  void AddJoinedString(const std::vector<StringRef>& strings,
                       StringRef separator);

  // Fill content into a buffer and returns the number of bytes stored.
  // The function allocates space for the buffer but does NOT take ownership.
  int WriteToBuffer(char** buffer);

  // Fill content into a string tensor, with the given new_shape. The new shape
  // must match the number of strings in this object. Caller relinquishes
  // ownership of new_shape. If 'new_shape' is nullptr, keep the tensor's
  // existing shape.
  void WriteToTensor(TfLiteTensor* tensor, TfLiteIntArray* new_shape);

  // Fill content into a string tensor. Set shape to {num_strings}.
  void WriteToTensorAsVector(TfLiteTensor* tensor);

 private:
  // Data buffer to store contents of strings, not including headers.
  std::vector<char> data_;
  // Offset of the starting index of each string in data buffer.
  std::vector<size_t> offset_;
  // Max length in number of characters that we permit the total
  // buffer containing the concatenation of all added strings to be.
  // For historical reasons this is limited to 32bit length. At this files
  // inception, sizes were represented using 32bit which forced an implicit cap
  // on the size of the buffer. When this was refactored to use size_t (which
  // could be 64bit) we enforce that the buffer remains at most 32bit length to
  // avoid a change in behavior.
  const size_t max_length_;
};

// Return num of strings in a String tensor.
int GetStringCount(const void* raw_buffer);
int GetStringCount(const TfLiteTensor* tensor);

// Get String pointer and length of index-th string in tensor.
// NOTE: This will not create a copy of string data.
StringRef GetString(const void* raw_buffer, int string_index);
StringRef GetString(const TfLiteTensor* tensor, int string_index);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_STRING_UTIL_H_

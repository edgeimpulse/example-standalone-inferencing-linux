/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tensorflow-lite/tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow-lite/tensorflow/lite/core/c/common.h"
#include "tensorflow-lite/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow-lite/tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow-lite/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow-lite/tensorflow/lite/kernels/internal/types.h"
#include "tensorflow-lite/tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dynamic_update_slice {

constexpr int kOperandTensor = 0;
constexpr int kUpdateTensor = 1;
constexpr int kStartIndicesTensor = 2;
constexpr int kOutputTensor = 0;

// TFLite DynamicUpdateSlice op follows the semantics of XLA DynamicUpdateSlice
// op. See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
// for details.
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* update;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kUpdateTensor, &update));
  const TfLiteTensor* start_indices;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartIndicesTensor,
                                          &start_indices));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // The shape of start_indices must be rank == 1, with dimension size equal to
  // the rank of operand.
  TF_LITE_ENSURE(context, NumDimensions(start_indices) == 1);
  TF_LITE_ENSURE(context,
                 SizeOfDimension(start_indices, 0) == NumDimensions(operand));

  // Update must be less than or equal to the operand size for each dimension to
  // avoid generating out-of-bounds update indices.
  TF_LITE_ENSURE(context, NumDimensions(update) == NumDimensions(operand));
  for (int i = 0; i < NumDimensions(operand); i++) {
    TF_LITE_ENSURE(context,
                   SizeOfDimension(update, i) <= SizeOfDimension(operand, i));
  }

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, update->type);
  TF_LITE_ENSURE_TYPES_EQ(context, start_indices->type, kTfLiteInt32);

  output->type = operand->type;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(operand->dims);
  return context->ResizeTensor(context, output, output_size);
}

// A helper function that converts a tensor index into a flat array index.
// Takes `start_indices` as an offset if not null.
int TensorIndexToFlat(const int* index, const int dims,
                      const RuntimeShape& shape,
                      const int* start_indices = nullptr) {
  int flat_index = index[0] + (start_indices ? start_indices[0] : 0);
  for (int i = 1; i < dims; i++) {
    flat_index = flat_index * shape.Dims(i) + index[i] +
                 (start_indices ? start_indices[i] : 0);
  }
  return flat_index;
}

// A helper function to compute the clamped start indices to ensure they are
// not out of bounds.
std::vector<int> ClampStartIndices(int input_dims, const int32_t* indices_data,
                                   const RuntimeShape& input_shape,
                                   const RuntimeShape& update_shape) {
  std::vector<int> clamped_start_indices(input_dims, 0);
  for (int i = 0; i < input_dims; i++) {
    clamped_start_indices[i] =
        std::min(std::max(0, indices_data[i]),
                 input_shape.Dims(i) - update_shape.Dims(i));
  }
  return clamped_start_indices;
}

template <typename T>
void DynamicUpdateSlice(const TfLiteTensor* input, const TfLiteTensor* update,
                        const TfLiteTensor* indice, TfLiteTensor* output) {
  const auto& input_shape = GetTensorShape(input);
  const auto& update_shape = GetTensorShape(update);
  const T* update_data = GetTensorData<T>(update);
  const int32_t* indices_data = GetTensorData<int32_t>(indice);
  T* output_data = GetTensorData<T>(output);

  const int input_dims = input_shape.DimensionsCount();
  // Computes the effective slice indices.
  // The clamped indices are gauranteed to >= 0 since update is less than or
  // equal to the operand size for each dimension.
  std::vector<int> clamped_start_indices =
      ClampStartIndices(input_dims, indices_data, input_shape, update_shape);

  // If the operation is not done in-place, copy the input data to the output.
  if (input->data.data != output->data.data) {
    memcpy(output->data.data, input->data.data, input->bytes);
  }

  // Update tensor has no elements. Skip.
  if (update_shape.FlatSize() == 0) {
    return;
  }

  std::vector<int> current_dim(input_dims, 0);
  // Overwrites update to output.
  do {
    int flat_update_index =
        TensorIndexToFlat(current_dim.data(), input_dims, update_shape);
    int flat_input_index =
        TensorIndexToFlat(current_dim.data(), input_dims, input_shape,
                          clamped_start_indices.data());
    output_data[flat_input_index] = update_data[flat_update_index];
  } while (NextIndex(input_dims,
                     reinterpret_cast<const int*>(update_shape.DimsData()),
                     current_dim.data()));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  const TfLiteTensor* update;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kUpdateTensor, &update));
  const TfLiteTensor* indice;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kStartIndicesTensor, &indice));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (operand->type) {
    case kTfLiteFloat32:
      DynamicUpdateSlice<float>(operand, update, indice, output);
      break;
    case kTfLiteBool:
      DynamicUpdateSlice<bool>(operand, update, indice, output);
      break;
    case kTfLiteInt8:
      DynamicUpdateSlice<int8_t>(operand, update, indice, output);
      break;
    case kTfLiteInt32:
      DynamicUpdateSlice<int32_t>(operand, update, indice, output);
      break;
    case kTfLiteInt64:
      DynamicUpdateSlice<int64_t>(operand, update, indice, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "DynamicUpdateSlice only currently supports "
                         "1-bit/8-bit/32-bit/64-bit integer or "
                         "float type, got %d.",
                         operand->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}
}  // namespace dynamic_update_slice

TfLiteRegistration* Register_DYNAMIC_UPDATE_SLICE() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 dynamic_update_slice::Prepare,
                                 dynamic_update_slice::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0,
                                 /*registration_external=*/nullptr,
                                 /*async_kernel=*/nullptr,
                                 kTfLiteInplaceOpInput0Shared};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

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
#include "tensorflow-lite/tensorflow/lite/core/async/c/task.h"

#include "tensorflow-lite/tensorflow/lite/core/async/c/types.h"
#include "tensorflow-lite/tensorflow/lite/core/async/task_internal.h"
#include "tensorflow-lite/tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow-lite/tensorflow/lite/core/c/common.h"

extern "C" {

TfLiteStatus TfLiteExecutionTaskSetBuffer(TfLiteExecutionTask* task,
                                          TfLiteIoType io_type,
                                          const char* tensor_signature_name,
                                          TfLiteBufferHandle handle) {
  if (task == nullptr || task->task == nullptr ||
      tensor_signature_name == nullptr)
    return kTfLiteError;
  return task->task->SetBufferHandle(io_type, tensor_signature_name, handle);
}

TfLiteStatus TfLiteExecutionTaskSetBufferByIndex(TfLiteExecutionTask* task,
                                                 int tensor_index,
                                                 TfLiteBufferHandle handle) {
  if (task == nullptr || task->task == nullptr) return kTfLiteError;
  return task->task->SetBufferHandle(tensor_index, handle);
}

TfLiteStatus TfLiteExecutionTaskSetSync(TfLiteExecutionTask* task,
                                        TfLiteIoType io_type,
                                        const char* tensor_signature_name,
                                        TfLiteSynchronization* sync) {
  if (task == nullptr || task->task == nullptr ||
      tensor_signature_name == nullptr)
    return kTfLiteError;
  return task->task->SetSynchronization(io_type, tensor_signature_name, sync);
}

TfLiteStatus TfLiteExecutionTaskSetSyncByIndex(TfLiteExecutionTask* task,
                                               int tensor_index,
                                               TfLiteSynchronization* sync) {
  if (task == nullptr || task->task == nullptr) return kTfLiteError;
  return task->task->SetSynchronization(tensor_index, sync);
}

TfLiteBufferHandle TfLiteExecutionTaskGetBufferByName(
    const TfLiteExecutionTask* task, TfLiteIoType io_type,
    const char* tensor_signature_name) {
  if (task == nullptr || task->task == nullptr ||
      tensor_signature_name == nullptr)
    return kTfLiteNullBufferHandle;
  return task->task->GetBufferHandle(io_type, tensor_signature_name);
}

TfLiteSynchronization* TfLiteExecutionTaskGetSyncByName(
    const TfLiteExecutionTask* task, TfLiteIoType io_type,
    const char* tensor_signature_name) {
  if (task == nullptr || task->task == nullptr ||
      tensor_signature_name == nullptr)
    return nullptr;
  return task->task->GetSynchronization(io_type, tensor_signature_name);
}

TfLiteBufferHandle TfLiteExecutionTaskGetBufferByIndex(
    const TfLiteExecutionTask* task, int tensor_index) {
  if (task == nullptr || task->task == nullptr) return kTfLiteNullBufferHandle;
  return task->task->GetBufferHandle(tensor_index);
}

TfLiteSynchronization* TfLiteExecutionTaskGetSyncByIndex(
    const TfLiteExecutionTask* task, int tensor_index) {
  if (task == nullptr || task->task == nullptr) return nullptr;
  return task->task->GetSynchronization(tensor_index);
}

void* TfLiteExecutionTaskGetDelegateExecutionData(
    const TfLiteExecutionTask* task, TfLiteAsyncKernel* kernel) {
  if (task == nullptr || task->task == nullptr) return nullptr;
  return task->task->GetDelegateExecutionData(kernel);
}

void TfLiteExecutionTaskSetDelegateExecutionData(TfLiteExecutionTask* task,
                                                 TfLiteAsyncKernel* kernel,
                                                 void* data) {
  if (task == nullptr || task->task == nullptr) return;
  task->task->SetDelegateExecutionData(kernel, data);
}

TfLiteStatus TfLiteExecutionTaskGetStatus(const TfLiteExecutionTask* task) {
  if (task == nullptr || task->task == nullptr) return kTfLiteError;
  return task->task->Status();
}

void TfLiteExecutionTaskSetStatus(TfLiteExecutionTask* task,
                                  TfLiteStatus status) {
  if (task == nullptr || task->task == nullptr) return;
  task->task->SetStatus(status);
}
}

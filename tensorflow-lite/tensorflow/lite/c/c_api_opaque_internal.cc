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
#include "tensorflow-lite/tensorflow/lite/c/c_api_opaque_internal.h"

#include <memory>
#include <unordered_map>
#include <utility>

#include "tensorflow-lite/tensorflow/lite/core/c/common.h"
#include "tensorflow-lite/tensorflow/lite/core/c/registration_external.h"
#include "tensorflow-lite/tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace internal {

namespace {

// Returns a dynamically allocated object; the caller is responsible for
// deallocating it using TfLiteRegistrationExternalDelete.
TfLiteRegistrationExternal* MakeRegistrationExternal(
    const TfLiteRegistration* registration, int node_index) {
  // We need to allocate a new TfLiteRegistrationExternal object and then
  // populate its state correctly, based on the contents in 'registration'.

  auto* registration_external = TfLiteRegistrationExternalCreate(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code),
      registration->custom_name, registration->version);

  registration_external->node_index = node_index;

  return registration_external;
}

}  // anonymous namespace

TfLiteRegistrationExternal*
CommonOpaqueConversionUtil::CachedObtainRegistrationExternal(
    RegistrationExternalsCache* registration_externals_cache,
    const TfLiteRegistration* registration, int node_index) {
  OpResolver::OpId op_id{registration->builtin_code, registration->custom_name,
                         registration->version};
  auto it = registration_externals_cache->find(op_id);
  if (it != registration_externals_cache->end()) {
    return it->second.get();
  }
  auto* registration_external =
      MakeRegistrationExternal(registration, node_index);
  registration_externals_cache->insert(
      it, std::make_pair(op_id, registration_external));

  return registration_external;
}

TfLiteRegistrationExternal*
CommonOpaqueConversionUtil::ObtainRegistrationExternal(
    TfLiteContext* context, const TfLiteRegistration* registration,
    int node_index) {
  auto* subgraph = static_cast<tflite::Subgraph*>(context->impl_);
  if (!subgraph->registration_externals_) {
    subgraph->registration_externals_ =
        std::make_shared<RegistrationExternalsCache>();
  }
  return CachedObtainRegistrationExternal(
      subgraph->registration_externals_.get(), registration, node_index);
}

}  // namespace internal
}  // namespace tflite

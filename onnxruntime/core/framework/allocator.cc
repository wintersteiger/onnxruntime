// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/utils.h"
#include <cstdlib>
#include <sstream>

namespace onnxruntime {

void* CPUAllocator::Alloc(size_t size) {
  return utils::DefaultAlloc(size);
}

void CPUAllocator::Free(void* p) {
  utils::DefaultFree(p);
}

const OrtMemoryInfo& CPUAllocator::Info() const { return *allocator_info_; }
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const OrtMemoryInfo& info) { return (out << info.ToString()); }

ORT_API_STATUS_IMPL(OrtCreateAllocatorInfo, _In_ const char* name1, OrtAllocatorType type, int id1,
                    OrtMemType mem_type1, _Out_ OrtMemoryInfo** out) {
  if (strcmp(name1, onnxruntime::CPU) == 0) {
    *out = new OrtMemoryInfo(name1, type, OrtDevice(), id1, mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA) == 0) {
    *out = new OrtMemoryInfo(
        name1, type, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, static_cast<OrtDevice::DeviceId>(id1)), id1,
        mem_type1);
  } else if (strcmp(name1, onnxruntime::CUDA_PINNED) == 0) {
    *out = new OrtMemoryInfo(
        name1, type, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, static_cast<OrtDevice::DeviceId>(id1)),
        id1, mem_type1);
  } else {
    return OrtCreateStatus(ORT_INVALID_ARGUMENT, "Specified device is not supported.");
  }
  return nullptr;
}

ORT_API(void, OrtReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo* p) { delete p; }

ORT_API_STATUS_IMPL(OrtMemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out) {
  *out = ptr->name;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtMemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out) {
  *out = ptr->id;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtMemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out) {
  *out = ptr->mem_type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtMemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out) {
  *out = ptr->type;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtCompareAllocatorInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2,
                    _Out_ int* out) {
  *out = (*info1 == *info2) ? 0 : -1;
  return nullptr;
}

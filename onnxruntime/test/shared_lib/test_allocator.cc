// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "test_fixture.h"

using namespace onnxruntime;

TEST_F(CApiTest, allocation_info) {
  OrtMemoryInfo *info1, *info2;
  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &info1));
  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &info2));
  int result;
  ORT_THROW_ON_ERROR(OrtCompareAllocatorInfo(info1, info2, &result));
  ASSERT_EQ(0, result);
  OrtReleaseMemoryInfo(info1);
  OrtReleaseMemoryInfo(info2);
}

TEST_F(CApiTest, DefaultAllocator) {
  Ort::AllocatorWithDefaultOptions default_allocator;
  char* p = (char*)default_allocator.Alloc(100);
  ASSERT_NE(p, nullptr);
  memset(p, 0, 100);
  default_allocator.Free(p);
  const OrtMemoryInfo* info1 = default_allocator.GetInfo();
  const OrtMemoryInfo* info2 = static_cast<OrtAllocator*>(default_allocator)->Info(default_allocator);
  int result;
  ORT_THROW_ON_ERROR(OrtCompareAllocatorInfo(info1, info2, &result));
  ASSERT_EQ(0, result);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The Ort C++ API is a header only wrapper around the Ort C API.
//
// The C++ API simplifies usage by returning values directly instead of error codes, throwing exceptions on errors
// and automatically releasing resources in the destructors.
//
// Each of the C++ wrapper classes holds only a pointer to the C internal object. Treat them like smart pointers.
// To create an empty object, pass 'nullptr' to the constructor (for example, Env e{nullptr};).
//
// Only move assignment between objects is allowed, there are no copy constructors. Some objects have explicit 'Clone'
// methods for this purpose.

#pragma once
#include "onnxruntime_c_api.h"
#include <cstddef>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace Ort {

using std::nullptr_t;

// All C++ methods that can fail will throw an exception of this type
struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

// This Macro is to make it easy to generate overloaded methods for all of the various OrtRelease* functions for every Ort* type
#define ORT_DEFINE_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { OrtRelease##NAME(ptr); }

ORT_DEFINE_RELEASE(MemoryInfo);
ORT_DEFINE_RELEASE(CustomOpDomain);
ORT_DEFINE_RELEASE(Env);
ORT_DEFINE_RELEASE(RunOptions);
ORT_DEFINE_RELEASE(Session);
ORT_DEFINE_RELEASE(SessionOptions);
ORT_DEFINE_RELEASE(TensorTypeAndShapeInfo);
ORT_DEFINE_RELEASE(TypeInfo);
ORT_DEFINE_RELEASE(Value);

template <typename T>
struct Base {
  Base() = default;
  Base(T* p) : p_{p} {
    if (!p) throw Ort::Exception("Allocation failure", ORT_FAIL);
  }
  ~Base() { OrtRelease(p_); }

  operator T*() { return p_; }
  operator const T*() const { return p_; }

  T* release() {
    T* p = p_;
    p_ = nullptr;
    return p;
  }

 protected:
  Base(const Base&) = delete;
  Base(Base&& v) noexcept : p_{v.p_} { v.p_ = nullptr; }
  void operator=(Base&& v) noexcept {
    OrtRelease(p_);
    p_ = v.p_;
    v.p_ = nullptr;
  }

  T* p_{};

  template <typename>
  friend struct Unowned;
};

template <typename T>
struct Unowned : T {
  Unowned(decltype(T::p_) p) : T{p} {}
  Unowned(Unowned&& v) : T{v.p_} {}
  ~Unowned() { this->p_ = nullptr; }
};

struct AllocatorWithDefaultOptions;
struct AllocatorInfo;
struct Env;
struct TypeInfo;
struct Value;

struct Env : Base<OrtEnv> {
  Env(nullptr_t) {}
  Env(OrtLoggingLevel default_logging_level, _In_ const char* logid);
  Env(OrtLoggingLevel default_logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);
  explicit Env(OrtEnv* p) : Base<OrtEnv>{p} {}
};

struct CustomOpDomain : Base<OrtCustomOpDomain> {
  explicit CustomOpDomain(nullptr_t) {}
  explicit CustomOpDomain(const char* domain);

  void Add(OrtCustomOp* op);
};

struct RunOptions : Base<OrtRunOptions> {
  RunOptions(nullptr_t) {}
  RunOptions();

  RunOptions& SetRunLogVerbosityLevel(int);
  int GetRunLogVerbosityLevel() const;

  RunOptions& SetRunLogSeverityLevel(int);
  int GetRunLogSeverityLevel() const;

  RunOptions& SetRunTag(const char* run_tag);
  const char* GetRunTag() const;

  RunOptions& EnableTerminate();
  RunOptions& DisableTerminate();
};

struct SessionOptions : Base<OrtSessionOptions> {
  explicit SessionOptions(nullptr_t) {}
  SessionOptions();
  explicit SessionOptions(OrtSessionOptions* p) : Base<OrtSessionOptions>{p} {}

  SessionOptions Clone() const;

  SessionOptions& SetThreadPoolSize(int session_thread_pool_size);
  SessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level);

  SessionOptions& EnableCpuMemArena();
  SessionOptions& DisableCpuMemArena();

  SessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_file);

  SessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);
  SessionOptions& DisableProfiling();

  SessionOptions& EnableMemPattern();
  SessionOptions& DisableMemPattern();

  SessionOptions& EnableSequentialExecution();
  SessionOptions& DisableSequentialExecution();

  SessionOptions& SetLogId(const char* logid);

  SessionOptions& Add(OrtCustomOpDomain* custom_op_domain);
};

struct Session : Base<OrtSession> {
  explicit Session(nullptr_t) {}
  Session(Env& env, const ORTCHAR_T* model_path, const SessionOptions& options);
  Session(Env& env, const void* model_data, size_t model_data_length, const SessionOptions& options);

  // Run that will allocate the output values
  std::vector<Value> Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
                         const char* const* output_names, size_t output_count);
  // Run for when there is a list of prealloated outputs
  void Run(const RunOptions& run_options, const char* const* input_names, Value* input_values, size_t input_count,
           const char* const* output_names, Value* output_values, size_t output_count);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;

  char* GetInputName(size_t index, OrtAllocator* allocator) const;
  char* GetOutputName(size_t index, OrtAllocator* allocator) const;

  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
};

struct TensorTypeAndShapeInfo : Base<OrtTensorTypeAndShapeInfo> {
  explicit TensorTypeAndShapeInfo(nullptr_t) {}
  explicit TensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* p) : Base<OrtTensorTypeAndShapeInfo>{p} {}

  ONNXTensorElementDataType GetElementType() const;
  size_t GetElementCount() const;

  size_t GetDimensionsCount() const;
  void GetDimensions(int64_t* values, size_t values_count) const;
  std::vector<int64_t> GetShape() const;
};

struct TypeInfo : Base<OrtTypeInfo> {
  explicit TypeInfo(nullptr_t) {}
  explicit TypeInfo(OrtTypeInfo* p) : Base<OrtTypeInfo>{p} {}

  Unowned<TensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;
  ONNXType GetONNXType() const;
};

struct Value : Base<OrtValue> {
  template <typename T>
  static Value CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len,
                            ONNXTensorElementDataType type);
  template <typename T>
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len);
  static Value CreateTensor(OrtAllocator* allocator, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type);

  static Value CreateMap(Value& keys, Value& values);
  static Value CreateSequence(std::vector<Value>& values);

  explicit Value(nullptr_t) {}
  explicit Value(OrtValue* p) : Base<OrtValue>{p} {}

  bool IsTensor() const;
  size_t GetCount() const;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
  Value GetValue(int index, OrtAllocator* allocator) const;

  size_t GetStringTensorDataLength() const;
  void GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const;

  template <typename T>
  T* GetTensorMutableData();

  TypeInfo GetTypeInfo() const;
  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;
};

struct AllocatorWithDefaultOptions {
  AllocatorWithDefaultOptions();

  operator OrtAllocator*() { return p_; }
  operator const OrtAllocator*() const { return p_; }

  void* Alloc(size_t size);
  void Free(void* p);

  const OrtMemoryInfo* GetInfo() const;

 private:
  OrtAllocator* p_{};
};

struct AllocatorInfo : Base<OrtMemoryInfo> {
  static AllocatorInfo CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);

  explicit AllocatorInfo(nullptr_t) {}
  AllocatorInfo(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);

  explicit AllocatorInfo(OrtMemoryInfo* p) : Base<OrtMemoryInfo>{p} {}
};

//
// Custom OPs (only needed to implement custom OPs)
//

struct CustomOpApi {
  CustomOpApi(const OrtCustomOpApi& api) : api_(api) {}

  template <typename T>  // T is only implemented for float, int64_t, and string
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value);
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info);
  size_t GetDimensionCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value);
  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value);

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info);
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input);
  size_t KernelContext_GetInputCount(const OrtKernelContext* context);
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index);
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context);
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count);

 private:
  const OrtCustomOpApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](OrtCustomOp* this_, const OrtCustomOpApi* api, const OrtKernelInfo* info) { return static_cast<TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetName(); };

    OrtCustomOp::GetExecutionProviderType = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetExecutionProviderType(); };

    OrtCustomOp::GetInputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](OrtCustomOp* this_) { return static_cast<TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](OrtCustomOp* this_, size_t index) { return static_cast<TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
  }

  // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
  const char* GetExecutionProviderType() const { return nullptr; }
};

}  // namespace Ort

#include "onnxruntime_cxx_inline.h"

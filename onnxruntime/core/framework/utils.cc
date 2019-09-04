// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"

#include <iomanip>

#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace utils {
void* DefaultAlloc(size_t size) {
  if (size <= 0) return nullptr;
  void* p;
  size_t alignment = MlasGetPreferredBufferAlignment();
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr) throw std::bad_alloc();
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0) throw std::bad_alloc();
#endif
  return p;
}

void DefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info) {
  return session_state.GetExecutionProviders().GetAllocator(allocator_info);
}

bool ProviderIsCpuBased(const std::string& provider_type) {
  return provider_type == onnxruntime::kCpuExecutionProvider ||
         provider_type == onnxruntime::kMklDnnExecutionProvider ||
         provider_type == onnxruntime::kNGraphExecutionProvider ||
         provider_type == onnxruntime::kNupharExecutionProvider ||
         provider_type == onnxruntime::kOpenVINOExecutionProvider ||
         provider_type == onnxruntime::kNnapiExecutionProvider;
}

common::Status AllocateHelper(const IExecutionProvider& execution_provider, const OrtDevice& device,
                              const Tensor& fetched_tensor, OrtValue& output_mlvalue) {
  auto allocator = execution_provider.GetAllocator(device.Id(), OrtMemTypeDefault);
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator");
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input && info.kci && info.kci->kernel_def->IsInputOnCpu(info.index);

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

static Status CopyMLValue(const DataTransferManager& data_transfer_mgr,
                          const MLValueCopyInfo& copy_info,
                          const OrtValue& source_mlvalue,
                          OrtValue& target_mlvalue) {
  if (copy_info.source_device == copy_info.target_device) {
    target_mlvalue = source_mlvalue;
    return Status::OK();
  }

  // This shouldn't be necessary. Edge case may be an unused input that has a mismatch between source and target
  // but as it's unused we have no allocator info (and don't want to allocate it as it's unused). Uncomment if needed.
  //  if (copy_info.allocation_provider == nullptr) {
  //  target_mlvalue = source_mlvalue;
  //  return Status::OK();
  //}

  // validate assumption that non-tensors are in fact CPU only
  assert(source_mlvalue.IsTensor());

  auto& source_tensor = source_mlvalue.Get<Tensor>();
  if (!target_mlvalue.IsAllocated()) {
    ORT_RETURN_IF_ERROR(utils::AllocateHelper(*copy_info.allocation_provider, copy_info.target_device,
                                              source_tensor, target_mlvalue));
  }

  Tensor* p_output_tensor = target_mlvalue.GetMutable<Tensor>();

  ORT_RETURN_IF_ERROR(data_transfer_mgr.CopyTensor(source_tensor, *p_output_tensor));

  return Status::OK();
}

static bool HaveCpuExecutionProvidersOnly(const ExecutionProviders& execution_providers) {
  for (const auto& execution_provider : execution_providers) {
    if (!ProviderIsCpuBased(execution_provider->Type())) {
      return false;
    }
  }

  return true;
}

// get the target device info for the node consuming each input provided in the feeds
static common::Status CalculateStaticCopyInfoForFeed(const SessionState& session_state,
                                                     const std::string& input_name,
                                                     MLValueCopyInfo& copy_info) {
  const auto& exec_providers = session_state.GetExecutionProviders();

  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));
  const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine

  if (node_info.p_node == nullptr) {
    // ignore dummy entry for an input that we didn't find a use of in the graph.
    return Status::OK();
  }

  copy_info.target_device = *node_info.device;

  const auto& required_provider_type = GetNodeInputProviderType(node_info);
  const auto* required_provider = exec_providers.Get(required_provider_type);
  copy_info.allocation_provider = required_provider;

  return Status::OK();
}

static common::Status CalculateStaticCopyInfoForFeeds(const SessionState& session_state,
                                                      const std::vector<std::string>& feed_names,
                                                      std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = feed_names.size(); idx < end; ++idx) {
    CalculateStaticCopyInfoForFeed(session_state, feed_names[idx], copy_info[idx]);
  }

  return Status::OK();
}

// get the source device info for the node producing each output that we will return in the fetches
static common::Status CalculateStaticCopyInfoForFetches(const SessionState& session_state,
                                                        const std::vector<std::string>& fetch_names,
                                                        std::vector<MLValueCopyInfo>& copy_info) {
  for (size_t idx = 0, end = fetch_names.size(); idx < end; ++idx) {
    const std::string& output_name = fetch_names[idx];

    std::vector<SessionState::NodeInfo> node_info_vec;
    ORT_RETURN_IF_ERROR(session_state.GetOutputNodeInfo(output_name, node_info_vec));
    const auto& node_info = node_info_vec.front();  // only one entry as only one node can produce a given output

    copy_info[idx].source_device = *node_info.device;
  }

  return Status::OK();
}

common::Status InitializeFeedFetchCopyInfo(const SessionState& session_state,
                                           FeedsFetchesManager& feeds_fetches_manager) {
  // if we only have CPU based EPs we can skip all the copy logic
  auto cpu_only = HaveCpuExecutionProvidersOnly(session_state.GetExecutionProviders());

  if (cpu_only) {
    feeds_fetches_manager.SetDeviceCopyChecks(DeviceCopyCheck::NoCopy, DeviceCopyCheck::NoCopy);
  } else {
    // setup all the static info about where the graph inputs and outputs are located
    auto info = feeds_fetches_manager.GetFeedsFetchesInfo();
    auto& feed_copy_info = feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo();
    auto& fetch_copy_info = feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo();
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFeeds(session_state, info.feed_names, feed_copy_info));
    ORT_RETURN_IF_ERROR(utils::CalculateStaticCopyInfoForFetches(session_state, info.output_names, fetch_copy_info));
  }

  return Status::OK();
}

// update the allocation_provider in the copy info based on the actual feeds
static bool FinalizeCopyInfoForFeeds(const std::vector<OrtDevice>& feed_locations,
                                     std::vector<MLValueCopyInfo>& copy_info) {
  ORT_ENFORCE(feed_locations.size() == copy_info.size());
  bool copy_needed = false;

  for (size_t i = 0, end = feed_locations.size(); i < end; ++i) {
    copy_info[i].source_device = feed_locations[i];

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
    }
  }

  return copy_needed;
}

static bool FinalizeCopyInfoForFetches(const SessionState& session_state,
                                       const std::vector<const OrtAllocatorInfo*>& fetch_alloc_info,
                                       std::vector<MLValueCopyInfo>& copy_info) {
  bool copy_needed = false;

  auto& execution_providers = session_state.GetExecutionProviders();
  const auto& cpu_execution_provider = *execution_providers.Get(onnxruntime::kCpuExecutionProvider);  // never null

  auto num_outputs = fetch_alloc_info.size();
  for (int i = 0; i < num_outputs; ++i) {
    const IExecutionProvider* provider = &cpu_execution_provider;
    const auto* alloc_info = fetch_alloc_info[i];

    if (alloc_info != nullptr) {
      copy_info[i].target_device = alloc_info->device;
      provider = execution_providers.Get(*alloc_info);
    }

    if (copy_info[i].source_device != copy_info[i].target_device) {
      copy_needed = true;
      copy_info[i].allocation_provider = provider;
    }
  }

  return copy_needed;
}

// Finalize the copy info using the OrtDevice and OrtAllocatorInfo for the feeds and fetches
// This can be used by control flow nodes prior to the execution of the overall graph.
common::Status FinalizeFeedFetchCopyInfo(const SessionState& session_state,
                                         FeedsFetchesManager& feeds_fetches_manager,
                                         const std::vector<OrtDevice>& feed_locations,
                                         const std::vector<const OrtAllocatorInfo*>& fetch_alloc_info) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return Status::OK();

  bool need_copy = FinalizeCopyInfoForFeeds(feed_locations, feeds_fetches_manager.GetMutableFeedsDeviceCopyInfo());
  DeviceCopyCheck input_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  need_copy = FinalizeCopyInfoForFetches(session_state, fetch_alloc_info,
                                         feeds_fetches_manager.GetMutableFetchesDeviceCopyInfo());
  DeviceCopyCheck output_copy = need_copy ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;

  feeds_fetches_manager.SetDeviceCopyChecks(input_copy, output_copy);

  return Status::OK();
}

// Finalize the copy info using the OrtValue instances for the feeds and fetches
static common::Status FinalizeFeedFetchCopyInfo(const SessionState& session_state,
                                                FeedsFetchesManager& feeds_fetches_manager,
                                                const std::vector<OrtValue>& feeds,
                                                std::vector<OrtValue>& fetches) {
  if (feeds_fetches_manager.GetDeviceCopyChecks().status == DeviceCopyCheck::NoCopy)
    return Status::OK();

  auto num_inputs = feeds.size();
  auto num_outputs = fetches.size();

  std::vector<OrtDevice> feed_locations(num_inputs);
  std::vector<const OrtAllocatorInfo*> fetch_alloc_info(num_outputs, nullptr);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& feed = feeds[i];
    if (feed.IsTensor()) {
      feed_locations[i] = feed.Get<Tensor>().Location().device;
    }
  }

  // create default instances if needed
  fetches.resize(num_outputs);

  for (int i = 0; i < num_outputs; ++i) {
    const auto& fetch = fetches[i];
    if (fetch.IsAllocated() && fetch.IsTensor()) {
      fetch_alloc_info[i] = &fetch.Get<Tensor>().Location();
    }
  }

  auto status = FinalizeFeedFetchCopyInfo(session_state, feeds_fetches_manager, feed_locations, fetch_alloc_info);

  return status;
}

static common::Status CopyInputsAcrossDevices(const std::vector<OrtValue>& orig_feeds,
                                              std::vector<OrtValue>& new_feeds,
                                              const std::vector<MLValueCopyInfo>& copy_info,
                                              const DataTransferManager& data_transfer_mgr) {
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(copy_info.size() == num_feeds);

  new_feeds.resize(num_feeds);

  for (size_t idx = 0; idx < num_feeds; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], orig_feeds[idx], new_feeds[idx]));
  }

  return Status::OK();
}

// public method to do a single copy. used by external partners
common::Status CopyOneInputAcrossDevices(const SessionState& session_state, const std::string& input_name,
                                         const OrtValue& orig_mlvalue, OrtValue& new_mlvalue) {
  if (!orig_mlvalue.IsTensor()) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  MLValueCopyInfo copy_info;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  ORT_RETURN_IF_ERROR(CalculateStaticCopyInfoForFeed(session_state, input_name, copy_info));
  copy_info.source_device = orig_mlvalue.Get<Tensor>().Location().device;

  return CopyMLValue(session_state.GetDataTransferMgr(), copy_info, orig_mlvalue, new_mlvalue);
}

static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               const std::vector<OrtValue>& fetches,
                                               std::vector<OrtValue>& user_fetches,
                                               const std::vector<MLValueCopyInfo>& copy_info) {
  auto num_outputs = fetches.size();
  user_fetches.resize(num_outputs);

  const auto& data_transfer_mgr = session_state.GetDataTransferMgr();

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    ORT_RETURN_IF_ERROR(CopyMLValue(data_transfer_mgr, copy_info[idx], fetches[idx], user_fetches[idx]));
  }

  return Status::OK();
}

static common::Status ExecuteGraphImpl(const SessionState& session_state,
                                       const FeedsFetchesManager& feeds_fetches_manager,
                                       const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                       const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                       bool sequential_execution, const bool& terminate_flag,
                                       const logging::Logger& logger) {
  std::unique_ptr<IExecutor> p_exec;
  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  const auto& device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  // see if we can skip copies due to the types of execution providers available
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators,
                                        logger));
  } else {
    const std::vector<OrtValue>* p_feeds = &feeds;
    std::vector<OrtValue>* p_fetches = &fetches;
    std::vector<OrtValue> device_feeds;
    std::vector<OrtValue> device_fetches;

    if (device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      const auto& feed_copy_info = feeds_fetches_manager.GetFeedsDeviceCopyInfo();
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(feeds, device_feeds, feed_copy_info,
                                                  session_state.GetDataTransferMgr()));
      p_feeds = &device_feeds;
    }

    auto num_outputs = fetches.size();
    const auto& fetch_copy_info = feeds_fetches_manager.GetFetchesDeviceCopyInfo();

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      // need intermediate fetches. use pre-allocated fetches where possible.
      device_fetches.reserve(num_outputs);

      for (int i = 0; i < num_outputs; ++i) {
        if (fetch_copy_info[i].source_device == fetch_copy_info[i].target_device && fetches[i].IsAllocated()) {
          device_fetches.push_back(fetches[i]);
        } else {
          // use temporary value
          device_fetches.push_back({});
        }
      }

      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, fetch_copy_info));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            bool sequential_execution, const bool& terminate_flag,
                            const logging::Logger& logger) {
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(session_state, feeds_fetches_manager));

  // finalize the copy info using the provided feeds and fetches. will update device_copy_checks in the background
  auto status = FinalizeFeedFetchCopyInfo(session_state, feeds_fetches_manager, feeds, fetches);
  ORT_RETURN_IF_ERROR(status);

  status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, {},
                            sequential_execution, terminate_flag, logger);

  return status;
}

common::Status ExecuteSubgraph(const SessionState& session_state, const FeedsFetchesManager& feeds_fetches_manager,
                               const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                               const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                               bool sequential_execution, const bool& terminate_flag, const logging::Logger& logger) {
  auto status = ExecuteGraphImpl(session_state, feeds_fetches_manager, feeds, fetches, fetch_allocators,
                                 sequential_execution, terminate_flag, logger);
  return status;
}

#if defined(DEBUG_NODE_INPUTS_OUTPUTS)
std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  return out << value.ToFloat();
}

std::ostream& operator<<(std::ostream& out, const MLFloat16& value) {
  return out << value.val;
}

template <typename T>
static void DumpTensor(const Tensor& tensor, const TensorShape& shape) {
  auto num_items = shape.Size();

  if (num_items == 0) {
    std::cout << "no data";
    return;
  }

  size_t num_dims = shape.NumDimensions();
  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }

  size_t row_size = num_items / num_rows;

  auto data = tensor.DataAsSpan<T>();

  auto print_val = [](const T& value) {
    if (std::is_floating_point_v<T>)
      std::cout << std::setprecision(8) << value;
    else
      std::cout << value;
  };

  for (int row = 0; row < num_rows; ++row) {
    print_val(data[row * row_size]);
    for (int i = 1; i < row_size; ++i) {
      std::cout << ", ";
      print_val(data[row * row_size + i]);
    }
    std::cout << "\n";
  }

  std::cout << std::endl;
}

void DumpNodeInputs(const OpKernelContext& context, const Node& node) {
  std::cout << "-----------\n";
  std::cout << node.OpType() << " node: " << node.Name() << "\n";

  const auto& input_defs = node.InputDefs();

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {
      std::cout << "Input " << i << " Name: " << input_defs[i]->Name();

      const auto* type = context.InputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Input<Tensor>(i);
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << " was missing data type\n";
      }
    } else {
      std::cout << "Input " << i << " is optional and was not provided.\n";
    }
  }
}

void DumpNodeOutputs(OpKernelContext& context, const Node& node, const SessionState& session_state) {
  std::cout << "-----------\n";
  const auto& output_defs = node.OutputDefs();

  const auto& execution_providers = session_state.GetExecutionProviders();
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {
      std::cout << "Output " << i << " Name: " << output_defs[i]->Name();

      const auto* type = context.OutputType(i);

      if (type) {
        if (type->IsTensorType()) {
          const auto& tensor = *context.Output<Tensor>(i);
          const auto data_type = tensor.DataType();
          const auto& shape = tensor.Shape();

          std::cout << " Shape: " << shape << "\n";

          // check tensor is on CPU before dumping it
          auto& tensor_location = tensor.Location();
          auto* provider = execution_providers.Get(tensor_location);
          if (!provider) {
            provider = cpu_execution_provider;
          }

          if (provider == cpu_execution_provider || tensor_location.mem_type == OrtMemTypeCPUOutput) {
            DispatchOnTensorType(data_type, DumpTensor, tensor, shape);
          } else {
            std::cout << " is not on CPU. Provider=" << provider->Type() << "\n";
          }
        } else {
          std::cout << " is non-tensor type.\n";
        }
      } else {
        // should never happen...
        std::cout << "missing data type\n";
      }
    } else {
      std::cout << "Output " << i << " is optional and was not produced.\n";
    }

    std::cout << std::endl;
  }
}
#endif

}  // namespace utils
}  // namespace onnxruntime

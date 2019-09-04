// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/utils.h"

#include "core/framework/execution_frame.h"
#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/sequential_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

#include "core/framework/tensorprotoutils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

/*
ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    OpSchema()
        .SetDoc("If conditional")
        .Input(0, "cond", "Condition for the if", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope. The return values in "
            "the `then_branch` and `else_branch` must be of the same shape and same "
            "data type.",
            "V",
            OpSchema::Variadic)
        .Attr(
            "then_branch",
            "Graph to run if condition is true. Has N outputs: values you wish to "
            "be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the else_branch.",
            AttributeProto::GRAPH)
        .Attr(
            "else_branch",
            "Graph to run if condition is false. Has N outputs: values you wish to"
            " be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the then_branch.",
            AttributeProto::GRAPH)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {"tensor(bool)"}, "Only bool"));
*/

ONNX_CPU_OPERATOR_KERNEL(If,
                         1,
                         KernelDefBuilder()
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         If);

struct If::Info {
  If::Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in)
      : subgraph{subgraph_in} {
    num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());
    num_outputs = static_cast<int>(node.OutputDefs().size());

    //auto& subgraph_inputs = subgraph.GetInputs();
    //auto num_subgraph_inputs = subgraph_inputs.size();
    auto& subgraph_outputs = subgraph.GetOutputs();
    auto num_subgraph_outputs = subgraph_outputs.size();

    //subgraph_input_names.reserve(num_subgraph_inputs);
    //for (int i = 0; i < num_subgraph_inputs; ++i) {
    //  subgraph_input_names.push_back(subgraph_inputs[i]->Name());
    //}

    if (num_subgraph_outputs != static_cast<size_t>(num_outputs)) {
      auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'If' node has ", num_outputs,
                                    " outputs which doesn't match the subgraph's ", num_subgraph_outputs, " outputs.");
      ORT_THROW(status);
    }

    subgraph_output_names.reserve(num_subgraph_outputs);
    for (size_t i = 0; i < num_subgraph_outputs; ++i) {
      auto& output = subgraph_outputs[i];
      subgraph_output_names.push_back(output->Name());
    }
  }

  const GraphViewer& subgraph;

  int num_implicit_inputs;
  int num_outputs;

  // std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;
};

class IfImpl {
 public:
  IfImpl(OpKernelContextInternal& context,
         const SessionState& session_state,
         const If::Info& info);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& ffm);

 private:
  Status AllocateOutputTensors();

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const If::Info& info_;

  std::unordered_map<std::string, const OrtValue*> implicit_inputs_;

  enum class AllocationType {
    Delayed,  // allocation of If output will be done by subgraph execution
    IfOutput
  };

  // track where the fetches provided to subgraph execution were allocated.
  std::vector<std::pair<AllocationType, OrtValue>> outputs_;
};

// Status IfImpl::CreateFeedsFetchesManager(std::unique_ptr<FeedsFetchesManager>& ffm) {
//  // we setup the FeedsFetchesInfo manually here as we need to skip implicit inputs that aren't in this subgraph
//  FeedsFetchesInfo ffi;
//
//  auto num_inputs = implicit_inputs_.size();
//  ffi.feed_names.reserve(num_inputs);
//  ffi.feeds_mlvalue_idxs.reserve(num_inputs);
//
//  auto& ort_value_name_idx_map = session_state_.GetOrtValueNameIdxMap();
//
//  // pass in implicit inputs as feeds.
//  for (auto& entry : implicit_inputs_) {
//    // prune to values that are in this subgraph as the implicit inputs cover both 'then' and 'else' subgraphs.
//    // alternatively we could track implicit inputs on a per-attribute basis in the node, but that
//    // would make that tracking a bit more complicated.
//    int idx;
//    if (ort_value_name_idx_map.GetIdx(entry.first, idx).IsOK()) {
//      ffi.feed_names.push_back(entry.first);
//      ffi.feeds_mlvalue_idxs.push_back(idx);
//    }
//  }
//
//  ffi.output_names = subgraph_output_names_;
//  ORT_RETURN_IF_ERROR(
//      FeedsFetchesInfo::MapNamesToMLValueIdxs(ffi.output_names, ort_value_name_idx_map, ffi.fetches_mlvalue_idxs));
//
//  ffm = std::make_unique<FeedsFetchesManager>(std::move(ffi));
//
//  return Status::OK();
//}

If::If(const OpKernelInfo& info) : OpKernel(info) {
  // make sure the required attributes are present even though we don't need it here.
  // The GraphProto attributes are loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("then_branch", &proto).IsOK());
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("else_branch", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);
}

If::~If() = default;

common::Status If::CreateFeedsFetchesManager(const SessionState& session_state,
                                             const std::string& attribute_name,
                                             const SessionState& subgraph_session_state) {
  const auto& node = Node();
  std::unique_ptr<If::Info>& info = attribute_name == "then_branch"
                                        ? then_info_
                                        : else_info_;

  ORT_ENFORCE(info == nullptr);
  info = std::make_unique<If::Info>(node, *subgraph_session_state.GetGraphViewer());

  // all inputs are implicit
  std::vector<std::string> feed_names;
  feed_names.reserve(info->num_implicit_inputs);

  const auto& subgraph_map = subgraph_session_state.GetOrtValueNameIdxMap();

  for (auto& entry : node.ImplicitInputDefs()) {
    // prune out entries that aren't in this subgraph as the 'then' and 'else' subgraphs are different
    // and implicit inputs covers both
    int idx;
    if (subgraph_map.GetIdx(entry->Name(), idx).IsOK()) {
      feed_names.push_back(entry->Name());
    }
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  auto status = FeedsFetchesManager::Create(feed_names, info->subgraph_output_names, subgraph_session_state, ffm);
  ORT_RETURN_IF_ERROR(status);

  status = utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm);
  ORT_RETURN_IF_ERROR(status);

  // default to CPU for all and override in FindDevicesForFeeds.
  // Use session state for this node and not the subgraph when looking for feed info
  std::vector<OrtDevice> feed_locations(feed_names.size());
  controlflow::detail::FindDevicesForFeeds(session_state, feed_names, feed_locations);

  // init to nullptr
  std::vector<const OrtAllocatorInfo*> fetch_locations;

  // we need the allocator info for each output from this node as we write directly into that
  const auto& outputs = node.OutputDefs();
  for (int i = 0, end = info->num_outputs; i < end; ++i) {
    const auto& alloc_info = controlflow::detail::FindAllocatorInfoForValue(session_state, outputs[i]->Name());
    fetch_locations.push_back(&alloc_info);
  }

  status = utils::FinalizeFeedFetchCopyInfo(subgraph_session_state, *ffm, feed_locations, fetch_locations);

  if (status.IsOK()) {
    if (attribute_name == "then_branch")
      then_feeds_fetches_manager_ = std::move(ffm);
    else
      else_feeds_fetches_manager_ = std::move(ffm);
  }

  return status;
}
Status If::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  auto condition = *ctx->Input<Tensor>(0)->Data<bool>();

  auto attribute = condition ? "then_branch" : "else_branch";
  auto* session_state = ctx_internal->SubgraphSessionState(attribute);
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for '", attribute, "' attribute.");

  const auto& info = condition ? then_info_ : else_info_;
  IfImpl impl{*ctx_internal, *session_state, *info};

  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  // create FeedsFetchesManager if needed and call IfImpl::Execute
  if (condition) {
    status = impl.Execute(*then_feeds_fetches_manager_);
  } else {
    status = impl.Execute(*else_feeds_fetches_manager_);
  }

  return status;
}

IfImpl::IfImpl(OpKernelContextInternal& context,
               const SessionState& session_state,
               const If::Info& info)
    : context_{context},
      session_state_{session_state},
      info_{info},
      implicit_inputs_{context_.GetImplicitInputs()} {
}

Status IfImpl::Initialize() {
  auto status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status IfImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  int index = 0;

  for (auto& graph_output : info_.subgraph.GetOutputs()) {
    auto* graph_output_shape = graph_output->Shape();
    if (!graph_output_shape) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                             graph_output->Name(), " did not.");
    }

    TensorShape output_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);

    // if size < 0 we have a symbolic dimension and need to use a temporary OrtValue in the subgraph execution
    if (output_shape.Size() < 0) {
      // we still need a value to put in the feeds we give to the execution frame, so just use an empty MLValue
      outputs_.push_back({AllocationType::Delayed, {}});
    } else {
      auto* tensor = context_.Output(index, output_shape);

      if (!tensor)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());

      outputs_.emplace_back(AllocationType::IfOutput, *context_.GetOutputMLValue(index));
    }

    ++index;
  }

  return Status::OK();
}

Status IfImpl::Execute(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  // pass in implicit inputs as feeds.
  // use the FeedsFetchesInfo as that has the pruned names
  auto& feed_names = ffm.GetFeedsFetchesInfo().feed_names;

  auto num_inputs = feed_names.size();
  std::vector<OrtValue> feeds;
  feeds.reserve(num_inputs);

  for (auto& feed_name : feed_names) {
    const auto* feed_mlvalue = implicit_inputs_[feed_name];
    ORT_ENFORCE(feed_mlvalue, "All implicit inputs should have OrtValue instances by now. ", feed_name, " did not.");

    feeds.push_back(*feed_mlvalue);
  }

  std::vector<OrtValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  fetches.reserve(info_.num_outputs);
  for (int i = 0; i < info_.num_outputs; ++i) {
    fetches.push_back(outputs_[i].second);

    if (outputs_[i].first == AllocationType::Delayed) {
      // functor to forward the allocation request from the subgraph to the If node's context so that the
      // allocation plan for the If node's output is used.
      fetch_allocators[i] = [this, i](const TensorShape& shape, OrtValue& ort_value) {
        // allocate
        auto* tensor = context_.Output(i, shape);

        if (!tensor) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for If output ", i);

        // return OrtValue for allocated tensor
        ort_value = *context_.GetOutputMLValue(i);
        return Status::OK();
      };
    }
  }

  status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, fetch_allocators,
                                  /*sequential_execution*/ true, context_.GetTerminateFlag(),
                                  context_.Logger());

  ORT_RETURN_IF_ERROR(status);

  return status;
}

}  // namespace onnxruntime

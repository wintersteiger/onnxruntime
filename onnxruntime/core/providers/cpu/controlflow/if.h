// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {
class SessionState;

class If final : public OpKernel, public controlflow::IControlFlowNode {
 public:
  If(const OpKernelInfo& info);
  //If(const OpKernelInfo& info) : OpKernel(info) {
  //  // make sure the required attributes are present even though we don't need it here.
  //  // The GraphProto attributes are loaded as a Graph instance by main Graph::Resolve,
  //  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  //  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  //  ONNX_NAMESPACE::GraphProto proto;
  //  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("then_branch", &proto).IsOK());
  //  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("else_branch", &proto).IsOK());
  //  ORT_IGNORE_RETURN_VALUE(proto);
  //}

  Status Compute(OpKernelContext* ctx) const override;

  common::Status CreateFeedsFetchesManager(const SessionState& session_state,
                                           const std::string& attribute_name,
                                           const SessionState& subgraph_session_state) override;

  struct Info;
  ~If();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(If);

 private:
  std::unique_ptr<Info> then_info_;
  std::unique_ptr<Info> else_info_;
  std::unique_ptr<FeedsFetchesManager> then_feeds_fetches_manager_;
  std::unique_ptr<FeedsFetchesManager> else_feeds_fetches_manager_;
};
}  // namespace onnxruntime

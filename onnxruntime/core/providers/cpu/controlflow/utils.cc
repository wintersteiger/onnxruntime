// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/utils.h"

#include <vector>
#include "core/common/common.h"
#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace controlflow {
namespace detail {

static const OrtAllocatorInfo& FindAllocatorInfoForValue(const OrtValueNameIdxMap& map,
                                                         const SequentialExecutionPlan& plan,
                                                         const std::string& name) {
  int idx = -1;
  auto status = map.GetIdx(name, idx);
  ORT_THROW_IF_ERROR(status);

  const auto& location = plan.GetLocation(idx);
  return location;
}

const OrtAllocatorInfo& FindAllocatorInfoForValue(const SessionState& session_state,
                                                  const std::string& name) {
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  return FindAllocatorInfoForValue(session_state.GetOrtValueNameIdxMap(), *exec_plan_ptr, name);
}

common::Status FindDevicesForFeeds(const SessionState& session_state,
                                   std::vector<std::string> feed_names,
                                   std::vector<OrtDevice>& feed_locations,
                                   size_t start_at) {
  // Currently this is simple. The control flow nodes run on CPU, and MemcpyTransformer makes sure all the
  // explicit and implicit inputs to a control flow node are on CPU.
  // In the future we will make things smarter to avoid unnecessary copies if the inputs are initially available on
  // a non-CPU device, and the subgraph wants to consume them on the same device.
  // At that point we will need some way to lookup where the explicit and implicit inputs
  // will be produced/available, which may be non trivial as they could be graph inputs, initializers
  // or node outputs, and there could be thousands of implicit inputs.

  // for (auto& name : feed_names) { in the future find device each feed is coming from  }

  feed_locations.resize(feed_names.size());

  const auto& map = session_state.GetOrtValueNameIdxMap();
  const auto* exec_plan_ptr = session_state.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr);

  for (size_t i = start_at, end = feed_names.size(); i < end; ++i) {
    const auto& location = FindAllocatorInfoForValue(map, *exec_plan_ptr, feed_names[i]);
    feed_locations[i] = location.device;
  }

  return Status::OK();
}

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime

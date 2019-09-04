// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
class Graph;

namespace controlflow {
namespace detail {

class IControlFlowNode {
 public:
  // helper to create the copy info upfront for the feeds and fetches used in each subgraph execution
  // TODO: Make this a formal interface that control flow nodes inherit from so we can formally check we're calling
  // this on a valid node?
  virtual common::Status CreateFeedsFetchesManager(const SessionState& session_state,
                                                   const std::string& attribute_name,
                                                   const SessionState& subgraph_session_state) = 0;
};

const OrtAllocatorInfo& FindAllocatorInfoForValue(const SessionState& session_state,
                                                  const std::string& name);

common::Status FindDevicesForFeeds(const SessionState& session_state,
                                   std::vector<std::string> feed_names,
                                   std::vector<OrtDevice>& feed_locations,
                                   size_t start_at = 0);

// helper to execute the subgraph by calling the Execute method of the provided implementation class with
// with the cached or newly created FeedsFetchesManager
template <typename TImpl>
common::Status SubgraphExecuteHelper(std::unique_ptr<FeedsFetchesManager>& cached_feeds_fetches_manager, TImpl& impl,
                                     std::once_flag& init_flag) {
  auto status = Status::OK();

  if (cached_feeds_fetches_manager) {
    // make it clear we don't update this instance when executing so there are no potential concurrency issues
    const FeedsFetchesManager* cached_ffm = &*cached_feeds_fetches_manager;
    status = impl.Execute(nullptr, cached_ffm);
  } else {
    // use a local instance until we know we're successful, and cache if it is.
    // doesn't matter if we do this multiple times. only matters that we only set cached_feeds_fetches_manager once.
    std::unique_ptr<FeedsFetchesManager> new_ffm;
    ORT_RETURN_IF_ERROR(impl.CreateFeedsFetchesManager(new_ffm));

    status = impl.Execute(&*new_ffm, nullptr);
    if (status.IsOK()) {
      std::call_once(init_flag, [&cached_feeds_fetches_manager, &new_ffm]() {
        cached_feeds_fetches_manager = std::move(new_ffm);
      });
    }
  }

  return status;
}

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/kiwipedia/kiwipedia_runtime.cc
 * \brief JSON runtime implementation for kiwipedia.
 */

#include <dmlc/parameter.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../file_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class kiwipedia_Runtime : public JSONRuntimeBase {
 public:
  explicit kiwipedia_Runtime(const std::string& symbol_name, const std::string& graph_json,
                             const ffi::Array<ffi::String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~kiwipedia_Runtime() override {
    if (matmul_so_handle_ != nullptr) {
      dlclose(matmul_so_handle_);
      matmul_so_handle_ = nullptr;
    }
    if (kv_cache_so_handle_ != nullptr) {
      dlclose(kv_cache_so_handle_);
      kv_cache_so_handle_ = nullptr;
    }
  }

  const char* kind() const final { return "kiwipedia"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
  }

  /*! \brief Run inference using kiwipedia JSON runtime. */
  void Run() override {
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      if (nodes_[nid].GetOpType() != "kernel") {
        continue;
      }

      const std::string op_name = nodes_[nid].GetOpName();
      if (op_name == "kiwipedia.matmul") {
        kiwipedia_matmul(nid);
      } else if (op_name == "kiwipedia.kv_cache_kernel") {
        kiwipedia_kv_cache_kernel(nid);
      } else {
        ICHECK(false) << "Unsupported kiwipedia kernel: " << op_name;
      }
    }
  }

 private:
  using MatmulFn =
      void (*)(std::vector<const DLTensor*>&, std::vector<int64_t>&, std::vector<int64_t>&);

  // Signature of libkv_cache_kernel.so:
  // extern "C" void kv_cache_kernel(data_entry, src_eid, cache_eid, token_pos_eid, out_eid)
  using KVCacheFn = void (*)(std::vector<const DLTensor*>&, int64_t, int64_t, int64_t, int64_t);

  void* matmul_so_handle_{nullptr};
  void* kv_cache_so_handle_{nullptr};
  MatmulFn matmul_fp_{nullptr};
  KVCacheFn kv_cache_fp_{nullptr};

  std::vector<int64_t> ShapeOfEntry(const JSONGraphNodeEntry& entry) const {
    auto shape_arr = nodes_[entry.id_].GetOpShape()[entry.index_];
    return std::vector<int64_t>(shape_arr.begin(), shape_arr.end());
  }

  void EnsureMatmulLoaded() {
    if (matmul_fp_ != nullptr) return;

    const char* env_path = std::getenv("KIWIPEDIA_MATMUL_SO");
    std::vector<const char*> candidates;
    if (env_path != nullptr && *env_path != '\0') candidates.push_back(env_path);
    candidates.push_back("libmatmul.so");
    candidates.push_back("./libmatmul.so");
    candidates.push_back("/home/giakoalan/tvm/src/runtime/contrib/kiwipedia/libmatmul.so");

    for (const char* path : candidates) {
      matmul_so_handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (matmul_so_handle_ == nullptr) continue;

      void* sym = dlsym(matmul_so_handle_, "matmul");
      if (sym == nullptr) {
        dlclose(matmul_so_handle_);
        matmul_so_handle_ = nullptr;
        continue;
      }

      matmul_fp_ = reinterpret_cast<MatmulFn>(sym);
      break;
    }

    ICHECK(matmul_fp_ != nullptr)
        << "Failed to load symbol 'matmul'. Set KIWIPEDIA_MATMUL_SO to libmatmul.so. "
        << "dlerror: " << dlerror();
  }

  void EnsureKVCacheLoaded() {
    if (kv_cache_fp_ != nullptr) return;

    const char* env_path = std::getenv("KIWIPEDIA_KV_CACHE_SO");
    std::vector<const char*> candidates;
    if (env_path != nullptr && *env_path != '\0') candidates.push_back(env_path);
    candidates.push_back("libkv_cache_kernel.so");
    candidates.push_back("./libkv_cache_kernel.so");
    candidates.push_back("/home/giakoalan/tvm/src/runtime/contrib/kiwipedia/libkv_cache_kernel.so");

    for (const char* path : candidates) {
      kv_cache_so_handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (kv_cache_so_handle_ == nullptr) continue;

      void* sym = dlsym(kv_cache_so_handle_, "kv_cache_kernel");
      if (sym == nullptr) {
        dlclose(kv_cache_so_handle_);
        kv_cache_so_handle_ = nullptr;
        continue;
      }

      kv_cache_fp_ = reinterpret_cast<KVCacheFn>(sym);
      break;
    }

    ICHECK(kv_cache_fp_ != nullptr)
        << "Failed to load symbol 'kv_cache_kernel'. "
        << "Set KIWIPEDIA_KV_CACHE_SO to libkv_cache_kernel.so. "
        << "dlerror: " << dlerror();
  }

  void kiwipedia_matmul(size_t nid) {
    const auto inputs = nodes_[nid].GetInputs();
    ICHECK_GE(inputs.size(), 2U) << "kiwipedia.matmul expects at least 2 inputs.";

    std::vector<int64_t> shape_a = ShapeOfEntry(inputs[0]);
    std::vector<int64_t> shape_b = ShapeOfEntry(inputs[1]);

    EnsureMatmulLoaded();
    matmul_fp_(data_entry_, shape_a, shape_b);
  }

  void kiwipedia_kv_cache_kernel(size_t nid) {
    const auto inputs = nodes_[nid].GetInputs();
    ICHECK_GE(inputs.size(), 3U)
        << "kiwipedia.kv_cache_kernel expects 3 inputs: src, old_cache, token_pos.";

    const int64_t src_eid = static_cast<int64_t>(EntryID(inputs[0]));
    const int64_t cache_eid = static_cast<int64_t>(EntryID(inputs[1]));
    const int64_t token_pos_eid = static_cast<int64_t>(EntryID(inputs[2]));
    const int64_t out_eid = static_cast<int64_t>(EntryID(static_cast<uint32_t>(nid), 0));

    EnsureKVCacheLoaded();
    kv_cache_fp_(data_entry_, src_eid, cache_eid, token_pos_eid, out_eid);
  }
};

ffi::Module kiwipediaRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                   const ffi::Array<ffi::String>& const_names) {
  auto n = ffi::make_object<kiwipedia_Runtime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.kiwipedia_runtime_create", kiwipediaRuntimeCreate)
      .def("ffi.Module.load_from_bytes.kiwipedia", JSONRuntimeBase::LoadFromBytes<kiwipedia_Runtime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

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

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../file_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

// user add
#include<stdio.h>
#include<dlfcn.h>
#include<stdlib.h>
#include<iostream>
// #include "matmul.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class kiwipedia_Runtime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The kiwipedia runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit kiwipedia_Runtime(const std::string& symbol_name, const std::string& graph_json,
                           const ffi::Array<ffi::String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* kind() const final { return "kiwipedia"; }

  /*!
   * \brief Initialize runtime. Create kiwipedia layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const ffi::Array<Tensor>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
  }

  ~kiwipedia_Runtime() {
    if (so_handle_ != nullptr) {
      dlclose(so_handle_);
      so_handle_ = nullptr;
    }
    if (kv_cache_so_handle_ != nullptr) {
      dlclose(kv_cache_so_handle_);
      kv_cache_so_handle_ = nullptr;
    }
  }

  /*! \brief Run inference using built engine. */
  void Run() override {

    arg_arr.clear();
    type_arr.clear();
    buffer_arr.clear();   

    // for(size_t i=0; i<nodes_.size(); i++){
      
    //   // shape
    //   // the following implementation is add operator's runtime, i need to implement matrix multiplication, which need to get the shape of m,n,o
    //   auto A_shape = nodes_[i].GetOpShape()[0][0];
    //   auto B_shape = nodes_[i].GetOpShape()[0][1];

    //   int64_t n = A_shape[A_shape.size() - 2];
    //   int64_t m = A_shape[A_shape.size() - 1];
    //   int64_t o = B_shape[B_shape.size() - 1];
    //   std::cout<<"N: "<<n<<", M: "<<m<<", O: "<<o<<std::endl;
    //   arg_arr.push_back(n);
    //   arg_arr.push_back(m);
    //   arg_arr.push_back(o);


    //   // dtype
    //   auto tmp = nodes_[i].GetOpDataType();
    //   auto node_type = tmp[0].code;
    //   auto node_type_byte = tmp[0].bits/8;
    //   type_arr.push_back({node_type, node_type_byte});


    //   auto buffer = (float*)malloc(node_arg * node_type_byte);
    //   buffer_arr.push_back(buffer);
    // }
    // for instance, we have matmul of [6, 1500, 384] multiply by [384, 384], they are equivalent with [x, n, m] multiply by [m, o]
    // std::cout<<"Run() starts here"<<std::endl;

    //A_shape = nodes_[0].GetOpShape()[0]; // vector<int64_t>
    //A_shape = data_entry_[0]->shape; // vector<int64_t>
    //B_shape = nodes_[1].GetOpShape()[0]; // vector<int64_t>
    //B_shape = data_entry_[1]->shape; // vector<int64_t>
    // std::cout<<A_shape<<std::endl;
    // std::cout<<B_shape<<std::endl;
    
    //int64_t x = A_shape[A_shape.size() - 3];
    //int64_t n = A_shape[A_shape.size() - 2];
    //int64_t m = A_shape[A_shape.size() - 1];
    //int64_t o = B_shape[B_shape.size() - 1];
    //std::cout<<"X: "<<x<<" N: "<<n<<", M: "<<m<<", O: "<<o<<std::endl;
    //arg_arr.push_back(x);
    //arg_arr.push_back(n);
    //arg_arr.push_back(m);
    //arg_arr.push_back(o);
    // for (size_t i = 0; i < input_nodes_.size(); ++i) {
    //   auto nid = input_nodes_[i];
    //   if (nodes_[nid].GetOpType() == "input") {
    //     memcpy(buffer_arr[nid], data_entry_[nid]->data, arg_arr[nid] * type_arr[nid].second);// data_entry's 0 and 1 is input A and B
    //   }
    // }

    // for (size_t i = 0; i < outputs_.size(); ++i) {
    //   auto tmp = outputs_[i];
    //   auto nid = tmp.id_;
    //   memcpy(buffer_arr[nid], data_entry_[nid]->data, arg_arr[nid] * type_arr[nid].second);
    // }

    // for(size_t i=0; i<nodes_.size(); i++){
    //   auto input = nodes_[i].GetInputs();
    //   std::cout << "node: " << i << std::endl;
    //   std::cout << "\top_type: " << nodes_[i].GetOpType() << std::endl;
    //   std::cout << "\tname:    " << nodes_[i].GetOpName() << std::endl;
    //   for(size_t j=0; j<input.size(); j++){
    //     std::cout << "\tinput " << j << " : " << input[j].id_ << std::endl;
    //   }
    // }

    // for(int i=0; i<data_entry_[0]->ndim; i++)
    //   std::cout << "shape " << i << " : " << *(data_entry_[0]->shape+i) << std::endl;
    
    for(size_t nid=0; nid<nodes_.size(); nid++){
      if(nodes_[nid].GetOpType() == "kernel"){
        if(nodes_[nid].GetOpName() == "kiwipedia.matmul")
          kiwipedia_matmul(nid);
	      else if (nodes_[nid].GetOpName() == "kiwipedia.add")
	        kiwipedia_add(nid);
        else if (nodes_[nid].GetOpName() == "kiwipedia.kv_cache_kernel")
          kiwipedia_kv_cache_kernel(nid);
        // 後續增加其他 OP
        else
          ICHECK(false) << "Unsupported kiwipedia kernel: " << nodes_[nid].GetOpName();
      }
    }
    // if we directly write data to data_entry_'s [2], then buffer_arr is not necessary

    // for (size_t i = 0; i < outputs_.size(); ++i) {  
    //   auto tmp = outputs_[i];
    //   auto nid = tmp.id_; // that constant is 2
    //   memcpy(data_entry_[nid]->data, buffer_arr[nid], arg_arr[nid] * type_arr[nid].second); // put back answer to [2]
    // }
  }

 private:
 
  using MatmulFn =
      void (*)(std::vector<const DLTensor*>&, std::vector<int64_t>&, std::vector<int64_t>&);

  using KVCacheFn =
      void (*)(std::vector<const DLTensor*>&, int64_t, int64_t, int64_t);

  // 一定要宣告成 class 成員
  void* so_handle_{nullptr};
  void* kv_cache_so_handle_{nullptr};

  MatmulFn matmul_fp_{nullptr};
  KVCacheFn kv_cache_fp_{nullptr};
  
  using AddFn =
    void (*)(std::vector<const DLTensor*>&,
             std::vector<int64_t>&,
             std::vector<int64_t>&);

  AddFn add_fp_{nullptr};
  void EnsureMatmulLoaded() {
    if (matmul_fp_) return;

    const char* env_path = std::getenv("KIWIPEDIA_MATMUL_SO");
    std::vector<const char*> candidates;
    if (env_path && *env_path) candidates.push_back(env_path);
    candidates.push_back("libmatmul.so");
    candidates.push_back("./libmatmul.so");
    candidates.push_back("/home/lewis56/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia/libmatmul.so");


    for (const char* p : candidates) {
      so_handle_ = dlopen(p, RTLD_NOW | RTLD_LOCAL);
      if (!so_handle_) continue;
      void* sym = dlsym(so_handle_, "matmul");
      if (!sym) {
        dlclose(so_handle_);
        so_handle_ = nullptr;
        continue;
      }
      matmul_fp_ = reinterpret_cast<MatmulFn>(sym);
      break;
    }

    ICHECK(matmul_fp_ != nullptr)
        << "Failed to load symbol 'matmul' from shared library. "
        << "Set KIWIPEDIA_MATMUL_SO to the absolute path of your RVV matmul .so. "
        << "dlerror: " << dlerror();
  }

void EnsureAddLoaded() {
    if (add_fp_) return;

    EnsureMatmulLoaded();  // 確保 so_handle_ 已經 dlopen 成功

    void* sym = dlsym(so_handle_, "add");
    if (sym) {
        add_fp_ = reinterpret_cast<AddFn>(sym);
    }

    ICHECK(add_fp_ != nullptr)
        << "Failed to load symbol 'add' from shared library. "
        << "Please make sure libmatmul.so exports function add.";
}

void EnsureKVCacheLoaded() {
  if (kv_cache_fp_) return;

  const char* env_path = std::getenv("KIWIPEDIA_KV_CACHE_SO");
  std::vector<const char*> candidates;
  if (env_path && *env_path) candidates.push_back(env_path);
  candidates.push_back("libkv_cache_kernel.so");
  candidates.push_back("./libkv_cache_kernel.so");
  candidates.push_back("/home/lewis56/tvm/src/runtime/contrib/kiwipedia/libkv_cache_kernel.so");

  for (const char* p : candidates) {
    kv_cache_so_handle_ = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    if (!kv_cache_so_handle_) continue;

    void* sym = dlsym(kv_cache_so_handle_, "kv_cache_kernel");
    if (!sym) {
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

  std::vector<int64_t> GetRuntimeShape(const DLTensor* t) {
    std::vector<int64_t> shape;
    for (int i = 0; i < t->ndim; ++i) {
      shape.push_back(t->shape[i]);
    }
    return shape;
  }
  // ---------------- 改寫這個：用 dlsym 叫進來 ----------------
  void kiwipedia_matmul(size_t idx) {
    EnsureMatmulLoaded();

    auto inputs = nodes_[idx].GetInputs();

    ICHECK_GE(inputs.size(), 2)
        << "kiwipedia.matmul expects at least 2 inputs";

    uint32_t a_eid = EntryID(inputs[0]);
    uint32_t b_eid = EntryID(inputs[1]);

    // 目前 node 的 output entry
    JSONGraphNodeEntry out_entry;
    out_entry.id_ = static_cast<uint32_t>(idx);
    out_entry.index_ = 0;
    out_entry.version_ = 0;
    uint32_t c_eid = EntryID(out_entry);

    const DLTensor* A = data_entry_[a_eid];
    const DLTensor* B = data_entry_[b_eid];
    const DLTensor* C = data_entry_[c_eid];

    std::vector<const DLTensor*> op_data;
    op_data.push_back(A);
    op_data.push_back(B);
    op_data.push_back(C);

    std::vector<int64_t> shapeA = GetRuntimeShape(A);
    std::vector<int64_t> shapeB = GetRuntimeShape(B);

    //std::cout << "[DEBUG][runtime] matmul node=" << idx
    //          << " a_eid=" << a_eid
    //          << " b_eid=" << b_eid
    //          << " c_eid=" << c_eid
    //          << std::endl;

    matmul_fp_(op_data, shapeA, shapeB);
  }

  void kiwipedia_add(size_t idx) {
    EnsureAddLoaded();

    auto inputs = nodes_[idx].GetInputs();

    ICHECK_GE(inputs.size(), 2)
        << "kiwipedia.add expects at least 2 inputs";

    uint32_t a_eid = EntryID(inputs[0]);
    uint32_t b_eid = EntryID(inputs[1]);

    JSONGraphNodeEntry out_entry;
    out_entry.id_ = static_cast<uint32_t>(idx);
    out_entry.index_ = 0;
    out_entry.version_ = 0;
    uint32_t c_eid = EntryID(out_entry);

    const DLTensor* A = data_entry_[a_eid];
    const DLTensor* B = data_entry_[b_eid];
    const DLTensor* C = data_entry_[c_eid];

    std::vector<const DLTensor*> op_data;
    op_data.push_back(A);
    op_data.push_back(B);
    op_data.push_back(C);

    std::vector<int64_t> shapeA = GetRuntimeShape(A);
    std::vector<int64_t> shapeB = GetRuntimeShape(B);

    //std::cout << "[DEBUG][runtime] add node=" << idx
    //          << " a_eid=" << a_eid
    //          << " b_eid=" << b_eid
    //          << " c_eid=" << c_eid
    //          << std::endl;

    add_fp_(op_data, shapeA, shapeB);
  }

  void kiwipedia_kv_cache_kernel(size_t idx) {
    EnsureKVCacheLoaded();

    auto inputs = nodes_[idx].GetInputs();

    ICHECK_GE(inputs.size(), 2)
        << "kiwipedia.kv_cache_kernel expects 2 inputs: old_cache, src";

    uint32_t cache_eid = EntryID(inputs[0]);
    uint32_t src_eid = EntryID(inputs[1]);

    JSONGraphNodeEntry out_entry;
    out_entry.id_ = static_cast<uint32_t>(idx);
    out_entry.index_ = 0;
    out_entry.version_ = 0;
    uint32_t out_eid = EntryID(out_entry);

    kv_cache_fp_(data_entry_,
                static_cast<int64_t>(cache_eid),
                static_cast<int64_t>(src_eid),
                static_cast<int64_t>(out_eid));
  }
  //void kiwipedia_matmul(size_t idx){
    // open shared library

    // void *handle = dlopen("/home/fre930727/tvm/libmatmul.so", RTLD_LAZY);
    // if (!handle) {
    //     printf("open lib error\n");
    //     return;
    // }

    // // define function pointer type
    // typedef void (*matmul_t)(std::vector<const DLTensor*>& , std::vector<int64_t>& , std::vector<int64_t>& );

    // // load symbol
    // matmul_t matmul = (matmul_t)dlsym(handle, "matmul");

    // if (!matmul) {
    //     printf("use function error\n");
    //     dlclose(handle);
    //     return;
    // }
      
      //if(data_entry_[0]->dtype.code == 0)
      //  typedef int TYPE;
      //else if(data_entry_[0]->dtype.code == 2)
      //  typedef float TYPE;
      //else;
      
      //typedef float TYPE;

      /*
      int size = (data_entry_[0]->dtype.bits/8);
      TYPE* out = (TYPE*)malloc(arg * size);*/

      // inputs 
      // std::vector<TYPE*> input_arr;
      // auto inputs = nodes_[idx].GetInputs();
      // for (size_t i = 0; i < inputs.size(); ++i) {
      //   auto nid = inputs[i].id_;
      //   input_arr.push_back((TYPE*)(buffer_arr[nid]));
      // }
      // auto out = (TYPE*)(buffer_arr[idx]);

      /*
      for (size_t i = 0; i < input_nodes_.size(); ++i) {
        auto nid = input_nodes_[i];
        if (nodes_[nid].GetOpType() == "input") {
          TYPE* tmp = (TYPE*)malloc(arg * size);
          memcpy(tmp, data_entry_[nid]->data, arg * size);
          input_arr.push_back(tmp);
        }
      }*/

      // add(input_arr[0], input_arr[1], out, arg_arr[idx]);
      //matmul(data_entry_, A_shape, B_shape);

      // outputs 
      /*
      for (size_t i = 0; i < outputs_.size(); ++i) {
        uint32_t eid = EntryID(outputs_[i]);
        memcpy(data_entry_[eid]->data, out, arg * size);
      }*/

      //free(out);
      //dlclose(handle);
    //}

    std::vector<int64_t> arg_arr;
    std::vector<std::pair<uint8_t, uint8_t>> type_arr;
    std::vector<void*> buffer_arr;
    std::vector<int64_t> A_shape;
    std::vector<int64_t> B_shape;
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

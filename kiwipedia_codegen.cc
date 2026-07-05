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
  * \file src/relax/backend/contrib/kiwipedia/codegen.cc
  * \brief Implementation of the kiwipedia JSON serializer.
  */
  #include <tvm/ffi/function.h>
  #include <tvm/ffi/reflection/registry.h>
  #include <tvm/ir/module.h>
  #include <tvm/ir/transform.h>
  // TODO(sunggg): add operator attribute when it's ready
  // #include <tvm/relax/attrs/nn.h>
  #include <tvm/relax/type.h>

  #include <memory>
  #include <string>
  #include <unordered_set>
  #include <vector>

  #include "../../../transform/utils.h"
  #include "../codegen_json/codegen_json.h"
  #include "../utils.h"

  namespace tvm {
  namespace relax {
  namespace contrib {

  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
  using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
  using OpAttrExtractor = backend::contrib::OpAttrExtractor;
  using JSONSerializer = backend::contrib::JSONSerializer;

  class kiwipediaJSONSerializer;

  static bool IsSupportedKiwipediaComposite(const std::string& name) {
  static const std::unordered_set<std::string> supported = {
      "kiwipedia.matmul",
      "kiwipedia.kv_cache_kernel",
  };
  return supported.count(name) != 0;
  }

  /*!
  * \brief Collect the constants and attributes from all operator calls in the body
  * of a "Composite" function.
  */
  class kiwipediaCollectFromCompositeFunctionBody : public ExprVisitor {
  public:
    explicit kiwipediaCollectFromCompositeFunctionBody(kiwipediaJSONSerializer* serializer)
        : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

    void VisitExpr_(const ConstantNode* constant_node) final;
    void VisitExpr_(const CallNode* call_node) final;

    void SetGenericAttributes(const CallNode* call_node) {
      OpAttrExtractor extractor(node_);
      const Object* attr_obj = call_node->attrs.get();
      extractor.Extract(const_cast<Object*>(attr_obj));
    }

    kiwipediaJSONSerializer* serializer_;
    /*! \brief Accumulated translated arguments. */
    std::vector<JSONGraphNodeEntry> args_;
    /*!
    * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
    * final JSONGraphNode however we don't yet know how many inputs that will have.
    */
    JSONGraphObjectPtr node_;
  };

  /*!
  * \brief Generates an kiwipedia Module from a relax expression by serializing the expression to a
  * json representation. kiwipedia is not required here because use of kiwipedia APIs is deferred until
  * runtime.
  */
  class kiwipediaJSONSerializer : public JSONSerializer {
  public:
    explicit kiwipediaJSONSerializer(ffi::Map<Constant, ffi::String> constant_names, ffi::Map<Var, Expr> bindings)
        : JSONSerializer(constant_names), bindings_(bindings) {}

    using JSONSerializer::VisitExpr_;

    std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) final {
      // The call must be to an inline "Composite" function
      const auto* fn_var = call_node->op.as<VarNode>();
      ICHECK(fn_var);
      const auto fn = Downcast<Function>(bindings_[ffi::GetRef<Var>(fn_var)]);

      auto opt_composite = fn->GetAttr<ffi::String>(attr::kComposite);
      ICHECK(opt_composite.has_value());
      std::string name = opt_composite.value();

      ICHECK(IsSupportedKiwipediaComposite(name))
        << "Unsupported kiwipedia composite: " << name
        << ". Please add runtime dispatch and kernel implementation first.";

      // Collect the constants and attributes of all operator calls inside the composite body.
      kiwipediaCollectFromCompositeFunctionBody collector(this);
      collector.VisitExpr(fn->body);

      // Capture the args to the "Composite" function as inputs for this node.
      std::vector<JSONGraphNodeEntry> inputs;
      for (const auto& arg : call_node->args) {
        auto res = VisitExpr(arg);
        inputs.insert(inputs.end(), res.begin(), res.end());
      }

      // Capture constants from the composite function body as additional inputs for this node.
      for (const auto& node : collector.args_) {
        inputs.emplace_back(node);
      }

      // Create the final node.
      auto node = std::make_shared<JSONGraphNode>(name,
                                                  /*op_type=*/"kernel", inputs,
                                                  /*num_output=*/1);

      // Transfer attributes from the collector's node to the final node.
      node->CaptureAttrs(*collector.node_);

      VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";

      return AddNode(node, ffi::GetRef<Expr>(call_node));
    }

  private:
    /*! \brief The bindings to look up composite functions. */
    ffi::Map<Var, Expr> bindings_;
  };

  void kiwipediaCollectFromCompositeFunctionBody::VisitExpr_(const ConstantNode* constant_node) {
    for (const auto& entry : serializer_->VisitExpr(ffi::GetRef<Constant>(constant_node))) {
      args_.emplace_back(entry);
    }
  }

  void kiwipediaCollectFromCompositeFunctionBody::VisitExpr_(const CallNode* call_node) {
    SetGenericAttributes(call_node);
    ExprVisitor::VisitExpr_(call_node);
  }

  /*!
  * \brief Create runtime modules for kiwipedia.
  * \param functions The extern functions to be compiled via kiwipedia
  * \return Runtime modules.
  */
  ffi::Array<ffi::Module> kiwipediaCompiler(ffi::Array<Function> functions,
                                          ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                          ffi::Map<Constant, ffi::String> constant_names) {
    ffi::Array<ffi::Module> compiled_functions;
    for (const auto& func : functions) {
      VLOG(1) << "kiwipedia partition:" << std::endl << func;
      kiwipediaJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
      serializer.serialize(func);
      std::string graph_json = serializer.GetJSON();
      VLOG(1) << "kiwipedia JSON:" << std::endl << graph_json;
      auto constant_names = serializer.GetConstantNames();
      const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.kiwipedia_runtime_create");
      auto func_name = GetExtSymbol(func);
      auto result = pf(func_name, graph_json, constant_names);
      tvm::ffi::Module mod = result.cast<tvm::ffi::Module>();
      compiled_functions.push_back(mod);
    }
    return compiled_functions;
  }

  TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef().def("relax.ext.kiwipedia", kiwipediaCompiler);
  }

  }  // namespace contrib
  }  // namespace relax
  }  // namespace tvm

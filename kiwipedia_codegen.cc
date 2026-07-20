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
  #include <tvm/relax/expr_functor.h>

  #include <memory>
  #include <string>
  #include <unordered_set>
  #include <vector>
  #include <iostream>

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
        if (arg.as<ShapeExprNode>()) {
          continue;
        }

        if (arg->struct_info_.as<ShapeStructInfoNode>()) {
          continue;
        }

        ICHECK(arg->struct_info_.as<TensorStructInfoNode>())
            << "kiwipedia only supports TensorStructInfo inputs, but got: "
            << arg->struct_info_;

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

  static bool IsShapeArg(const Expr& expr) {
    if (expr.as<ShapeExprNode>()) {
      return true;
    }

    if (expr->struct_info_.defined() &&
        expr->struct_info_.as<ShapeStructInfoNode>()) {
      return true;
    }

    return false;
  }

  static bool IsKiwipediaCodegenFunc(const Function& func) {
    auto opt_codegen = func->GetAttr<ffi::String>("Codegen");
    if (opt_codegen.has_value() && opt_codegen.value() == "kiwipedia") {
      return true;
    }

    auto opt_composite = func->GetAttr<ffi::String>(attr::kComposite);
    if (opt_composite.has_value()) {
      std::string name = opt_composite.value();
      return IsSupportedKiwipediaComposite(name);
    }

    return false;
  }

  class RemoveKiwipediaShapeParamsMutator : public ExprMutator {
  public:
    explicit RemoveKiwipediaShapeParamsMutator(
        std::unordered_set<const GlobalVarNode*> kiwipedia_global_funcs)
        : kiwipedia_global_funcs_(kiwipedia_global_funcs) {}

    using ExprMutator::VisitExpr_;

    Expr VisitExpr_(const FunctionNode* func_node) final {
      Function old_func = ffi::GetRef<Function>(func_node);

      bool should_clean = IsKiwipediaCodegenFunc(old_func);

      bool old_inside = inside_kiwipedia_func_;
      inside_kiwipedia_func_ = old_inside || should_clean;

      ffi::Array<Var> new_params;
      for (const Var& param : old_func->params) {
        if (should_clean &&
            param->struct_info_.defined() &&
            param->struct_info_.as<ShapeStructInfoNode>()) {
          continue;
        }

        new_params.push_back(param);
      }

      Expr new_body = this->VisitExpr(old_func->body);

      inside_kiwipedia_func_ = old_inside;

      Function new_func = Function(
          new_params,
          new_body,
          old_func->ret_struct_info,
          old_func->is_pure,
          old_func->attrs,
          old_func->span
      );

      new_func.CopyOnWrite()->struct_info_ = old_func->struct_info_;

      return new_func;
    }

    Expr VisitExpr_(const CallNode* call_node) final {
      Expr new_op = this->VisitExpr(call_node->op);

      bool call_to_kiwipedia_global = false;
      if (const auto* gv = call_node->op.as<GlobalVarNode>()) {
        call_to_kiwipedia_global =
            kiwipedia_global_funcs_.count(gv) != 0;
      }

      bool should_clean_args =
          inside_kiwipedia_func_ || call_to_kiwipedia_global;

      ffi::Array<Expr> new_args;
      for (const Expr& arg : call_node->args) {
        if (should_clean_args && IsShapeArg(arg)) {
          continue;
        }

        new_args.push_back(this->VisitExpr(arg));
      }

      Call new_call = Call(
          new_op,
          new_args,
          call_node->attrs,
          call_node->sinfo_args,
          call_node->span
      );

      new_call.CopyOnWrite()->struct_info_ = call_node->struct_info_;

      return new_call;
    }

  private:
    std::unordered_set<const GlobalVarNode*> kiwipedia_global_funcs_;
    bool inside_kiwipedia_func_{false};
  };

  tvm::transform::Pass RemoveKiwipediaShapeParams() {
    auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
      std::unordered_set<const GlobalVarNode*> kiwipedia_global_funcs;

      for (const auto& kv : mod->functions) {
        GlobalVar gv = Downcast<GlobalVar>(kv.first);
        BaseFunc base_func = kv.second;

        if (const auto* func_node = base_func.as<FunctionNode>()) {
          Function func = ffi::GetRef<Function>(func_node);

          if (IsKiwipediaCodegenFunc(func)) {
            kiwipedia_global_funcs.insert(gv.get());
          }
        }
      }

      RemoveKiwipediaShapeParamsMutator mutator(kiwipedia_global_funcs);

      IRModule new_mod = mod;

      for (const auto& kv : mod->functions) {
        GlobalVar gv = Downcast<GlobalVar>(kv.first);
        BaseFunc base_func = kv.second;

        if (const auto* func_node = base_func.as<FunctionNode>()) {
          Function func = ffi::GetRef<Function>(func_node);
          Function new_func = Downcast<Function>(mutator.VisitExpr(func));

          new_mod.CopyOnWrite()->Add(gv, new_func, true);
        }
      }

      return new_mod;
    };

    return tvm::transform::CreateModulePass(
        pass_func,
        0,
        "RemoveKiwipediaShapeParams",
        {}
    );
  }
  /*
  class RemoveShapeParamsMutator : public ExprMutator {
  public:
    explicit RemoveShapeParamsMutator() = default;

    Expr VisitExpr_(const FunctionNode* func_node) final {
      Function old_func = ffi::GetRef<Function>(func_node);

      // 1. 只保留 TensorStructInfo 的 function params
      ffi::Array<Var> new_params;
      std::unordered_set<const VarNode*> removed_shape_params;

      for (const Var& param : old_func->params) {
        if (param->struct_info_.as<ShapeStructInfoNode>()) {
          removed_shape_params.insert(param.get());
          continue;
        }

        new_params.push_back(param);
      }

      // 2. 處理 function body，把 call args 裡對應的 shape var 移掉
      Expr new_body = this->VisitExpr(old_func->body);

      // 3. 重建 Function
      Function new_func = Function(
          new_params,
          new_body,
          old_func->ret_struct_info,
          old_func->is_pure,
          old_func->attrs,
          old_func->span
      );

      return new_func;
    }

    Expr VisitExpr_(const CallNode* call_node) final {
      Expr new_op = this->VisitExpr(call_node->op);

      ffi::Array<Expr> new_args;
      for (const Expr& arg : call_node->args) {
        // 移除 R.Shape(...)
        if (arg.as<ShapeExprNode>()) {
          continue;
        }

        // 移除 tir_vars: R.Shape(...)
        if (arg->struct_info_.as<ShapeStructInfoNode>()) {
          continue;
        }

        new_args.push_back(this->VisitExpr(arg));
      }

      Call new_call = Call(
          new_op,
          new_args,
          call_node->attrs,
          call_node->sinfo_args,
          call_node->span
      );

      // 保留原本 call 的 struct_info，避免 JSONSerializer 看不到 output shape
      new_call.CopyOnWrite()->struct_info_ = call_node->struct_info_;

      return new_call;
    }
  };

  static Function RemoveShapeParams(Function func) {
    RemoveShapeParamsMutator mutator;
    return Downcast<Function>(mutator.VisitExpr(func));
  }
  */
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

      auto new_constant_names = serializer.GetConstantNames();
      const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.kiwipedia_runtime_create");

      auto func_name = GetExtSymbol(func);
      auto result = pf(func_name, graph_json, new_constant_names);

      tvm::ffi::Module mod = result.cast<tvm::ffi::Module>();
      compiled_functions.push_back(mod);
    }

    return compiled_functions;
  }

  TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef()
        .def("relax.ext.kiwipedia", kiwipediaCompiler)
        .def("relax.transform.RemoveKiwipediaShapeParams", RemoveKiwipediaShapeParams);
  }

  }  // namespace contrib
  }  // namespace relax
  }  // namespace tvm

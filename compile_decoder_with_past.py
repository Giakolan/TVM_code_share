import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx  # Correct import path
from tvm.relax.dpl import is_op, wildcard
from tvm.contrib import cc
from tvm import tir


def compile_model(onnx_path, target="llvm"):
    # 1. Load ONNX model
    onnx_model = onnx.load(onnx_path) 
    # 2. Convert to Relax IR (updated API)

    #mod = from_onnx(onnx_model, {"input_features": (1, 80, 3000)})# give input shape of both encoder and decoder, make them static. Somer op does not support dynamic shape

    #mod = from_onnx(onnx_model, {"input_ids": (1, 1), "encoder_hidden_states": (1, 1500, 384)})# give input shape of both encoder and decoder, make them static. Somer op does not support dynamic shape
    
    batch_size = tir.Var("batch_size", "int64")
    decoder_seq_len = tir.Var("decoder_seq_len", "int64")
    past_decoder_seq_len = tir.Var("past_decoder_seq_len", "int64")
    encoder_seq_len = tir.Var("encoder_seq_len", "int64")

    num_layers = 4
    num_heads = 6
    head_dim = 64

    shape_dict = {
        "input_ids": (batch_size, decoder_seq_len),
    }

    for i in range(num_layers):
        shape_dict[f"past_key_values.{i}.decoder.key"] = (batch_size, num_heads, past_decoder_seq_len, head_dim)
        shape_dict[f"past_key_values.{i}.decoder.value"] = (batch_size, num_heads, past_decoder_seq_len, head_dim)
        shape_dict[f"past_key_values.{i}.encoder.key"] = (batch_size, num_heads, encoder_seq_len, head_dim)
        shape_dict[f"past_key_values.{i}.encoder.value"] = (batch_size, num_heads, encoder_seq_len, head_dim)

    mod = from_onnx(onnx_model, shape_dict)
    #mod=tvm.relax.transform.BindSymbolicVars({"batch_size":1, "encoder_sequence_length_out": 1500})(mod)

    #print("===== After from_onnx decoder_with_past =====")
    #mod.show()	

    patterns = [
        ("kiwipedia.kv_cache_kernel", is_op("relax.concat")(wildcard())),
        ("kiwipedia.matmul", is_op("relax.matmul")(wildcard(), wildcard())),
    ]


    #patterns = [("tensorrt.add", is_op("relax.add")(wildcard(), wildcard()))]

    '''
    annotate_codegen: 不要 Merge 相鄰的 OP，一個 OP 一個 Relax function
    bind_constants: 綁定常數，如果前面 from_onnx 的 keep_params_in_input=False(預設) 這裡要設成 bind_constants=False
                         如果前面 from_onnx 的 keep_params_in_input=True		這裡要設成 bind_constants=True(預設)
    '''
    #print("===== After LegalizeOps =====")
    #mod.show()

    mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
    #print("===== BEFORE RemoveKiwipediaShapeParams =====")
    #mod.show()
    #mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
    #mod = relax.transform.FuseOpsByPattern(patterns)(mod)
    #mod.show()
    #print("===== After FuseOpsByPattern =====")
    #mod.show()
    #mod = relax.transform.LambdaLift()(mod)
    #print("===== After LambdaLift =====")
    #mod.show()
    #mod = relax.transform.MergeCompositeFunctions()(mod)
    #mod.show()
    remove_shape_params = tvm.get_global_func(
        "relax.transform.RemoveKiwipediaShapeParams"
    )()

    mod = remove_shape_params(mod)

    print("===== AFTER RemoveKiwipediaShapeParams =====")
    mod.show()

    mod = relax.transform.RunCodegen()(mod)
    #mod.show()
    #print("===== After RunCodegen =====")
    #mod.show()

    # 3. Apply mandatory passes
    seq = tvm.ir.transform.Sequential([
        relax.transform.LegalizeOps(),
        #relax.transform.FoldConstant(),
        #relax.transform.DeadCodeElimination()
    ])
    mod = seq(mod)

    # Check if output IRModule is well-formed. 
    #assert relax.analysis.well_formed(mod)
    # 4. Build
    ex = relax.build(mod, target)
    
    # 5. Save
    output_path = onnx_path.replace(".onnx", ".so")
    ex.export_library(output_path)
    return output_path

# Compile both encoder and decoder
#encoder_so = compile_model("encoder_model.onnx", target="llvm")
#decoder_so = compile_model("decoder_model.onnx", target="llvm")
decoder_so = compile_model("decoder_with_past_model.onnx", target="llvm")

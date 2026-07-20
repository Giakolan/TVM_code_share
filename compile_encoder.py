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
	batch_size = tir.Var("batch_size", "int64")
	mod = from_onnx(onnx_model, {"input_features": (batch_size, 80, 3000)})# give input shape of both encoder and decoder, make them static. Somer op does not support dynamic shape

	#mod = from_onnx(onnx_model, {"input_ids": (1, 1), "encoder_hidden_states": (1, 1500, 384)})# give input shape of both encoder and decoder, make them static. Somer op does not support dynamic shape
	
	#mod = from_onnx(onnx_model)
	#mod=tvm.relax.transform.BindSymbolicVars({"batch_size":1, "encoder_sequence_length_out": 1500})(mod)


	print("===== After from_onnx =====")
	mod.show()

	#patterns = [("kiwipedia.matmul", is_op("relax.matmul")(wildcard(), wildcard()))]
	patterns = [
	    ("kiwipedia.matmul", is_op("relax.matmul")(wildcard(), wildcard()))
	]
	#patterns = [("tensorrt.add", is_op("relax.add")(wildcard(), wildcard()))]

	'''
	annotate_codegen: 不要 Merge 相鄰的 OP，一個 OP 一個 Relax function
	bind_constants: 綁定常數，如果前面 from_onnx 的 keep_params_in_input=False(預設) 這裡要設成 bind_constants=False
						 如果前面 from_onnx 的 keep_params_in_input=True		這裡要設成 bind_constants=True(預設)
	'''
	mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
	#mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
	#mod = relax.transform.FuseOpsByPattern(patterns)(mod)
	mod.show()



	#mod = relax.transform.MergeCompositeFunctions()(mod)
	#mod.show()



	mod = relax.transform.RunCodegen()(mod)
	mod.show()

	# 3. Apply mandatory passes
	seq = tvm.ir.transform.Sequential([
		relax.transform.LegalizeOps(),
		relax.transform.FoldConstant(),
		relax.transform.DeadCodeElimination()
	])
	mod = seq(mod)

	# Check if output IRModule is well-formed. 
	#assert relax.analysis.well_formed(mod)
	# 4. Build
	ex = relax.build(mod, target)
	
	# 5. Save
	output_path = onnx_path.replace(".onnx", ".so")
	#ex.export_library(output_path)
	return output_path

# Compile both encoder and decoder
encoder_so = compile_model("encoder_model.onnx", target="llvm")
#decoder_so = compile_model("decoder_model.onnx", target="llvm -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+a,+f,+d,+c -vector-width=128")


import tvm
from tvm import relax
from tvm.script.parser import relax as R
import numpy as np

import time

import onnx
from tvm.relax.frontend import onnx as tvm_onnx

from tvm import meta_schedule as ms
from tvm.meta_schedule.relax_integration import tune_relax 

from tvm.relax.dpl import is_op, wildcard   # use to patterns

from tvm.contrib import cc  # use to riscv_fcompile

onnx_model = onnx.load("/home/fre930727/resnet-18-ONNX/onnx/model.onnx")

''' 
設定模型 input 的維度與型態，沒有設定會是 dynamic shape (N, 3, 224, 224) 
keep_params_in_input: params 是否要當成 input，預設 False 會當成全域常數
'''
mod = tvm_onnx.from_onnx(onnx_model, shape_dict= ({"data":(1, 3, 224, 224)}), dtype_dict="float32", keep_params_in_input=False)
mod.show()

patterns = [("kiwipedia.matmul", is_op("relax.matmul")(wildcard(), wildcard()))]

'''
annotate_codegen: 不要 Merge 相鄰的 OP，一個 OP 一個 Relax function
bind_constants: 綁定常數，如果前面 from_onnx 的 keep_params_in_input=False(預設) 這裡要設成 bind_constants=False
                         如果前面 from_onnx 的 keep_params_in_input=True        這裡要設成 bind_constants=True(預設)
'''
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod.show()

mod = relax.transform.RunCodegen()(mod)
mod.show()

# Check if output IRModule is well-formed. 
assert relax.analysis.well_formed(mod)

# Define your target hardware
target= tvm.target.Target("c")

# Build and prepare VM.
ex = relax.build(mod, target, params={})

ex.export_library("/home/fre930727/tvm_rvv_matmul_byoc/tutorial/codegen_model_resnet18.so")



# ================================ inference ================================

# Define your device.
dev = tvm.cpu()

ex = tvm.runtime.load_module("/home/fre930727/tvm_rvv_matmul_byoc/tutorial/codegen_model_resnet18.so")
vm = relax.VirtualMachine(ex, dev)


data_np = np.random.rand(1, 3, 224, 224).astype(np.float32)
data = tvm.runtime.tensor(data_np, dev)

out = vm["main"](data)
out = out.numpy()
print(np.argmax(out))

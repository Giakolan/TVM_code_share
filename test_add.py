import numpy as np
import tvm
from tvm import relax
from tvm.relax.dpl import is_op, wildcard
from tvm.script import relax as R


@tvm.script.ir_module
class AddModule:
    @R.function
    def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")):
        with R.dataflow():
            z = R.add(x, y)
            R.output(z)
        return z


patterns = [
    ("kiwipedia.add", is_op("relax.add")(wildcard(), wildcard())),
]

mod = AddModule

print("===== Original IR =====")
mod.show()

mod = relax.transform.FuseOpsByPattern(patterns)(mod)
print("===== After FuseOpsByPattern =====")
mod.show()

mod = relax.transform.MergeCompositeFunctions()(mod)
print("===== After MergeCompositeFunctions =====")
mod.show()

mod = relax.transform.RunCodegen()(mod)
print("===== After RunCodegen =====")
mod.show()

ex = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

x = np.array([1, 2, 3, 4], dtype="float32")
y = np.array([10, 20, 30, 40], dtype="float32")

out = vm["main"](tvm.runtime.tensor(x), tvm.runtime.tensor(y))
print(out.numpy())

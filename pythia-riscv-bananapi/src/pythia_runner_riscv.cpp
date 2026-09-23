#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>


using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::runtime::Tensor;


// ------------------------------------------------------------
// Read exact-size binary file
// ------------------------------------------------------------
template <typename T>
bool ReadBinary(
    const char* path,
    std::vector<T>* data,
    size_t count
) {
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        std::printf(
            "FAIL: cannot open %s\n",
            path
        );
        return false;
    }

    const std::streamsize size =
        file.tellg();

    const std::streamsize expected =
        static_cast<std::streamsize>(
            count * sizeof(T)
        );

    if (size != expected) {
        std::printf(
            "FAIL: wrong file size: %s\n"
            "  got      = %ld bytes\n"
            "  expected = %ld bytes\n",
            path,
            static_cast<long>(size),
            static_cast<long>(expected)
        );

        return false;
    }

    file.seekg(0, std::ios::beg);

    data->resize(count);

    if (!file.read(
            reinterpret_cast<char*>(
                data->data()
            ),
            expected)) {

        std::printf(
            "FAIL: read error: %s\n",
            path
        );

        return false;
    }

    return true;
}


int main() {
    // --------------------------------------------------------
    // Paths
    // --------------------------------------------------------

    const char* model_path =
        "model/pythia_70m_riscv.so";

    const char* input_ids_path =
        "test_data/input_ids.bin";

    const char* attention_mask_path =
        "test_data/attention_mask.bin";

    const char* reference_path =
        "test_data/pytorch_logits.bin";


    // --------------------------------------------------------
    // Pythia static shapes
    // --------------------------------------------------------

    constexpr int64_t BATCH = 1;
    constexpr int64_t SEQ = 16;
    constexpr int64_t VOCAB = 50304;

    constexpr size_t INPUT_COUNT =
        BATCH * SEQ;

    constexpr size_t LOGIT_COUNT =
        BATCH * SEQ * VOCAB;


    try {
        std::printf(
            "========================================\n"
            "Pythia-70M RISC-V QEMU inference\n"
            "========================================\n"
        );


        // ----------------------------------------------------
        // 1. Read reference data
        // ----------------------------------------------------

        std::vector<int64_t> input_ids_data;
        std::vector<int64_t> attention_mask_data;
        std::vector<float> pytorch_logits;

        if (!ReadBinary(
                input_ids_path,
                &input_ids_data,
                INPUT_COUNT)) {
            return 1;
        }

        if (!ReadBinary(
                attention_mask_path,
                &attention_mask_data,
                INPUT_COUNT)) {
            return 1;
        }

        if (!ReadBinary(
                reference_path,
                &pytorch_logits,
                LOGIT_COUNT)) {
            return 1;
        }

        std::printf(
            "PASS: test data loaded\n"
        );


        // ----------------------------------------------------
        // 2. Load exported Relax executable
        // ----------------------------------------------------

        std::printf(
            "Loading Pythia artifact...\n"
        );

        Module executable =
            Module::LoadFromFile(
                model_path
            );

        std::printf(
            "PASS: artifact loaded\n"
        );


        // ----------------------------------------------------
        // 3. VMExecutable -> VirtualMachine
        // ----------------------------------------------------

        auto load_vm_opt =
            executable->GetFunction(
                "vm_load_executable"
            );

        if (!load_vm_opt.has_value()) {
            std::printf(
                "FAIL: vm_load_executable not found\n"
            );
            return 2;
        }

        Module vm =
            load_vm_opt.value()()
                .cast<Module>();

        std::printf(
            "PASS: Relax VM created\n"
        );


        // ----------------------------------------------------
        // 4. Initialize CPU VM
        //
        // device_type = 1 = kDLCPU
        // device_id   = 0
        // allocator   = 2 = pooled
        // ----------------------------------------------------

        auto vm_init_opt =
            vm->GetFunction(
                "vm_initialization"
            );

        if (!vm_init_opt.has_value()) {
            std::printf(
                "FAIL: vm_initialization not found\n"
            );
            return 3;
        }

        vm_init_opt.value()(
            1,
            0,
            2
        );

        std::printf(
            "PASS: VM initialized\n"
        );


        // ----------------------------------------------------
        // 5. Build input tensors
        // ----------------------------------------------------

        DLDevice dev;
        dev.device_type = kDLCPU;
        dev.device_id = 0;

        DLDataType int64_dtype;
        int64_dtype.code = kDLInt;
        int64_dtype.bits = 64;
        int64_dtype.lanes = 1;

        Tensor input_ids =
            Tensor::Empty(
                tvm::ffi::Shape(
                    {BATCH, SEQ}
                ),
                int64_dtype,
                dev
            );

        Tensor attention_mask =
            Tensor::Empty(
                tvm::ffi::Shape(
                    {BATCH, SEQ}
                ),
                int64_dtype,
                dev
            );

        input_ids.CopyFromBytes(
            input_ids_data.data(),
            INPUT_COUNT *
                sizeof(int64_t)
        );

        attention_mask.CopyFromBytes(
            attention_mask_data.data(),
            INPUT_COUNT *
                sizeof(int64_t)
        );

        std::printf(
            "PASS: input tensors created\n"
        );


        // ----------------------------------------------------
        // 6. Get Pythia main()
        // ----------------------------------------------------

        auto main_opt =
            vm->GetFunction(
                "main"
            );

        if (!main_opt.has_value()) {
            std::printf(
                "FAIL: VM main not found\n"
            );
            return 4;
        }

        std::printf(
            "\nRunning full Pythia inference...\n"
        );


        // ----------------------------------------------------
        // 7. FULL INFERENCE
        //
        // main()
        //   -> Relax VM
        //   -> Kiwipedia
        //   -> libmatmul_rvv.so
        //   -> RVV matmul
        // ----------------------------------------------------

        auto start =
            std::chrono::steady_clock::now();

        Tensor output =
            main_opt.value()(
                input_ids,
                attention_mask
            ).cast<Tensor>();

        auto end =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                end - start
            ).count();

        std::printf(
            "PASS: Pythia inference returned\n"
        );

        std::printf(
            "Inference wall time: %.6f seconds\n",
            seconds
        );


        // ----------------------------------------------------
        // 8. Copy logits back
        // ----------------------------------------------------

        std::vector<float> byoc_logits(
            LOGIT_COUNT
        );

        output.CopyToBytes(
            byoc_logits.data(),
            LOGIT_COUNT *
                sizeof(float)
        );

        std::printf(
            "PASS: logits copied\n"
        );


        // ----------------------------------------------------
        // 9. Compare all logits
        // ----------------------------------------------------

        double sum_diff = 0.0;
        float max_diff = 0.0f;
        size_t max_diff_index = 0;

        for (size_t i = 0;
             i < LOGIT_COUNT;
             ++i) {

            float diff =
                std::fabs(
                    byoc_logits[i]
                    - pytorch_logits[i]
                );

            sum_diff += diff;

            if (diff > max_diff) {
                max_diff = diff;
                max_diff_index = i;
            }
        }

        double mean_diff =
            sum_diff /
            static_cast<double>(
                LOGIT_COUNT
            );


        // ----------------------------------------------------
        // 10. Last-token argmax
        // ----------------------------------------------------

        const size_t last_token_offset =
            (SEQ - 1) * VOCAB;

        int pytorch_next = 0;
        int byoc_next = 0;

        float pytorch_best =
            pytorch_logits[
                last_token_offset
            ];

        float byoc_best =
            byoc_logits[
                last_token_offset
            ];

        for (int i = 1;
             i < VOCAB;
             ++i) {

            float ref =
                pytorch_logits[
                    last_token_offset + i
                ];

            float got =
                byoc_logits[
                    last_token_offset + i
                ];

            if (ref > pytorch_best) {
                pytorch_best = ref;
                pytorch_next = i;
            }

            if (got > byoc_best) {
                byoc_best = got;
                byoc_next = i;
            }
        }


        // ----------------------------------------------------
        // Results
        // ----------------------------------------------------

        std::printf(
            "\n========================================\n"
            "Correctness result\n"
            "========================================\n"
        );

        std::printf(
            "logit count       : %ld\n",
            static_cast<long>(
                LOGIT_COUNT
            )
        );

        std::printf(
            "max diff          : %.9f\n",
            max_diff
        );

        std::printf(
            "mean diff         : %.9f\n",
            mean_diff
        );

        std::printf(
            "max diff index    : %ld\n",
            static_cast<long>(
                max_diff_index
            )
        );

        std::printf(
            "PyTorch next token: %d\n",
            pytorch_next
        );

        std::printf(
            "BYOC next token   : %d\n",
            byoc_next
        );


        if (pytorch_next != byoc_next) {
            std::printf(
                "\nFAIL: next-token mismatch\n"
            );

            return 5;
        }


        std::printf(
            "\n========================================\n"
            "Pythia-70M RISC-V full inference PASS\n"
            "========================================\n"
        );

        return 0;

    } catch (const std::exception& e) {
        std::printf(
            "\nFAIL with exception:\n%s\n",
            e.what()
        );

        return 10;
    }
}

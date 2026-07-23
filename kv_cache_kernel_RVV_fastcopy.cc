#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace {

int64_t NumElements(const DLTensor* t) {
  int64_t n = 1;
  for (int i = 0; i < t->ndim; ++i) {
    n *= t->shape[i];
  }
  return n;
}

int64_t DTypeBytes(const DLTensor* t) {
  return static_cast<int64_t>(t->dtype.bits / 8) * static_cast<int64_t>(t->dtype.lanes);
}

int64_t ReadScalarInt64(const DLTensor* t) {
  if (t->dtype.code == kDLInt && t->dtype.bits == 64) {
    return *static_cast<int64_t*>(t->data);
  }
  if (t->dtype.code == kDLInt && t->dtype.bits == 32) {
    return static_cast<int64_t>(*static_cast<int32_t*>(t->data));
  }
  if (t->dtype.code == kDLUInt && t->dtype.bits == 64) {
    return static_cast<int64_t>(*static_cast<uint64_t*>(t->data));
  }
  if (t->dtype.code == kDLUInt && t->dtype.bits == 32) {
    return static_cast<int64_t>(*static_cast<uint32_t*>(t->data));
  }
  throw std::runtime_error("token_pos must be int32/int64/uint32/uint64 scalar tensor");
}

/*!
 * \brief Copy bytes with RVV when compiling on a RISC-V vector target.
 *
 * x86 / non-RVV platform:
 *   FastCopy -> std::memcpy
 *
 * RISC-V RVV platform:
 *   FastCopy -> vsetvl + vle8 + vse8
 *
 * 使用 byte-level copy 的原因：
 *   KV cache 可能是 int8 / fp16 / float32。
 *   對 cache update 而言，這裡只是搬資料，不需要理解數值型別。
 */
void FastCopy(uint8_t* dst, const uint8_t* src, size_t nbytes) {
#if defined(__riscv_vector)
  size_t i = 0;
  while (i < nbytes) {
    size_t vl = __riscv_vsetvl_e8m8(nbytes - i);
    vuint8m8_t v = __riscv_vle8_v_u8m8(src + i, vl);
    __riscv_vse8_v_u8m8(dst + i, v, vl);
    i += vl;
  }
#else
  std::memcpy(dst, src, nbytes);
#endif
}

}  // namespace

extern "C" void kv_cache_kernel(std::vector<const DLTensor*>& data_entry, int64_t src_eid,
                                int64_t cache_eid, int64_t token_pos_eid, int64_t out_eid) {
  const DLTensor* src = data_entry.at(static_cast<size_t>(src_eid));
  const DLTensor* old_cache = data_entry.at(static_cast<size_t>(cache_eid));
  const DLTensor* token_pos_tensor = data_entry.at(static_cast<size_t>(token_pos_eid));
  const DLTensor* out = data_entry.at(static_cast<size_t>(out_eid));

  if (src == nullptr || old_cache == nullptr || token_pos_tensor == nullptr || out == nullptr) {
    throw std::runtime_error("kv_cache_kernel received null DLTensor");
  }

  const int64_t elem_bytes = DTypeBytes(src);
  if (elem_bytes != DTypeBytes(old_cache) || elem_bytes != DTypeBytes(out)) {
    throw std::runtime_error("src/cache/out dtype size mismatch");
  }

  const int64_t token_pos = ReadScalarInt64(token_pos_tensor);
  const int64_t src_elems = NumElements(src);
  const int64_t cache_elems = NumElements(old_cache);
  const int64_t out_elems = NumElements(out);

  if (cache_elems != out_elems) {
    throw std::runtime_error("old_cache and out_cache must have the same number of elements");
  }
  if (src_elems <= 0) {
    throw std::runtime_error("src must not be empty");
  }

  auto* out_bytes = static_cast<uint8_t*>(out->data);
  const auto* cache_bytes = static_cast<const uint8_t*>(old_cache->data);
  const auto* src_bytes = static_cast<const uint8_t*>(src->data);

  // Functional semantics: out_cache = old_cache first.
  FastCopy(out_bytes, cache_bytes, static_cast<size_t>(cache_elems * elem_bytes));

  // Case A: old_cache = [batch, max_seq_len, ...], src = [batch, ...]
  if (old_cache->ndim == src->ndim + 1 && old_cache->ndim >= 2 &&
      old_cache->shape[0] == src->shape[0]) {
    const int64_t batch = src->shape[0];
    const int64_t max_seq_len = old_cache->shape[1];
    if (token_pos < 0 || token_pos >= max_seq_len) {
      throw std::runtime_error("token_pos out of range for [batch, max_seq_len, ...] cache");
    }

    const int64_t slice_elems_per_batch = src_elems / batch;
    for (int64_t b = 0; b < batch; ++b) {
      const int64_t src_offset = b * slice_elems_per_batch;
      const int64_t dst_offset = (b * max_seq_len + token_pos) * slice_elems_per_batch;

      FastCopy(out_bytes + dst_offset * elem_bytes, src_bytes + src_offset * elem_bytes,
               static_cast<size_t>(slice_elems_per_batch * elem_bytes));
    }
    return;
  }

  // Case B: old_cache = [max_seq_len, ...], src = [...]
  if (cache_elems % src_elems != 0) {
    throw std::runtime_error("cache_elems must be divisible by src_elems");
  }

  const int64_t max_seq_len = cache_elems / src_elems;
  if (max_seq_len <= 0 || token_pos < 0 || token_pos >= max_seq_len) {
    throw std::runtime_error("token_pos out of range for flat/[max_seq_len, ...] cache");
  }

  const int64_t dst_offset = token_pos * src_elems;
  FastCopy(out_bytes + dst_offset * elem_bytes, src_bytes,
           static_cast<size_t>(src_elems * elem_bytes));
}

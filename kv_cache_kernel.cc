#include <dlpack/dlpack.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

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

void CheckSameDTypeSize(const DLTensor* a, const DLTensor* b, const DLTensor* c) {
  const int64_t bytes = DTypeBytes(a);
  if (bytes != DTypeBytes(b) || bytes != DTypeBytes(c)) {
    throw std::runtime_error("kv_cache_kernel dtype size mismatch");
  }
}

}  // namespace

extern "C" void kv_cache_kernel(std::vector<const DLTensor*>& data_entry, int64_t cache_eid,
                                int64_t src_eid, int64_t out_eid) {
  const DLTensor* old_cache = data_entry.at(static_cast<size_t>(cache_eid));
  const DLTensor* src = data_entry.at(static_cast<size_t>(src_eid));
  const DLTensor* out = data_entry.at(static_cast<size_t>(out_eid));

  if (old_cache == nullptr || src == nullptr || out == nullptr) {
    throw std::runtime_error("kv_cache_kernel received null DLTensor");
  }

  CheckSameDTypeSize(old_cache, src, out);

  if (old_cache->ndim != src->ndim || old_cache->ndim != out->ndim) {
    throw std::runtime_error("kv_cache concat expects old_cache/src/out to have same ndim");
  }

  if (old_cache->ndim != 4) {
    throw std::runtime_error("kv_cache concat currently expects 4D tensors");
  }

  // Expected shapes:
  // old_cache: [batch, 6, past_decoder_seq_len, 64]
  // src:       [batch, 6, decoder_seq_len, 64]
  // out:       [batch, 6, past_decoder_seq_len + decoder_seq_len, 64]
  const int64_t batch = old_cache->shape[0];
  const int64_t heads = old_cache->shape[1];
  const int64_t past_len = old_cache->shape[2];
  const int64_t head_dim = old_cache->shape[3];
  const int64_t src_len = src->shape[2];

  if (src->shape[0] != batch || src->shape[1] != heads || src->shape[3] != head_dim) {
    throw std::runtime_error("kv_cache src shape mismatch");
  }

  if (out->shape[0] != batch || out->shape[1] != heads || out->shape[2] != past_len + src_len ||
      out->shape[3] != head_dim) {
    throw std::runtime_error("kv_cache out shape mismatch");
  }

  const int64_t elem_bytes = DTypeBytes(old_cache);
  const int64_t expected_out_elems = NumElements(old_cache) + NumElements(src);
  if (NumElements(out) != expected_out_elems) {
    throw std::runtime_error("kv_cache output element count mismatch");
  }

  auto* out_bytes = static_cast<uint8_t*>(out->data);
  const auto* cache_bytes = static_cast<const uint8_t*>(old_cache->data);
  const auto* src_bytes = static_cast<const uint8_t*>(src->data);

  const int64_t old_row_elems = past_len * head_dim;
  const int64_t src_row_elems = src_len * head_dim;
  const int64_t out_row_elems = (past_len + src_len) * head_dim;

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t bh = b * heads + h;

      const int64_t old_offset = bh * old_row_elems;
      const int64_t src_offset = bh * src_row_elems;
      const int64_t out_offset = bh * out_row_elems;

      std::memcpy(out_bytes + out_offset * elem_bytes, cache_bytes + old_offset * elem_bytes,
                  static_cast<size_t>(old_row_elems * elem_bytes));

      std::memcpy(out_bytes + (out_offset + old_row_elems) * elem_bytes,
                  src_bytes + src_offset * elem_bytes,
                  static_cast<size_t>(src_row_elems * elem_bytes));
    }
  }
}
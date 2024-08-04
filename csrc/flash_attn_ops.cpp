#include "kernels.h"
#include "pybind11_kernel_helpers.h"

namespace {

pybind11::bytes create_params(
    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,
    float softcap,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int window_size_left,
    int window_size_right,

    int is_causal,
    int is_bf16
) {
  return gpu_ops::PackDescriptor(gpu_ops::MHAParams{
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,

    q_row_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,

    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_head_stride,

    b,
    h,
    h_k,
    d,
    d_rounded,
    softmax_scale,
    softcap,

    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,

    window_size_left,
    window_size_right,

    is_causal,
    is_bf16
  });
}


pybind11::dict FlashAttnRegistrations() {
  pybind11::dict dict;
  dict["run_mha_fwd"] =
      gpu_ops::EncapsulateFunction(gpu_ops::run_mha_fwd_j);
  dict["run_mha_bwd"] =
      gpu_ops::EncapsulateFunction(gpu_ops::run_mha_bwd_j);
  return dict;
}

PYBIND11_MODULE(_jax_flash_attn, m) {
  m.def("get_flash_attn_registrations", &FlashAttnRegistrations);
  m.def("create_params", create_params);
}
} // namespace

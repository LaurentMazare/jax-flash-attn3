#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"
#include "flash_bwd_launch_template.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::bfloat16_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::bfloat16_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::bfloat16_t, 256>(params, stream);
            }
        } else {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::half_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::half_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::half_t, 256>(params, stream);
            }
        }
    } else {
        // if (params.d == 64) {
        //     run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
        // } else if (params.d == 128) {
        //     run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
        // } else if (params.d == 256) {
        //     run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
        // }        
    }
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  if (!params.is_bf16) {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::half_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::half_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::half_t, 128>(params, stream);
    }
  } else {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::bfloat16_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::bfloat16_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::bfloat16_t, 128>(params, stream);
    }
  }
}


extern "C" void rust_fn(void *stream, void **buffers, const char *opaque, std::size_t opaque_len);

// Wrapper around a rust function that can be called via the C++ ABI.
void cpp_wrap(cudaStream_t stream,
              void **buffers,
              const char *opaque,
              std::size_t opaque_len) {
  rust_fn((void*)stream, buffers, opaque, opaque_len);
}

// Export the pointer to the C++ wrapper.
extern "C" void* cpp_wrap_ptr() {
  return (void*)cpp_wrap;
}

extern "C" void run_mha_f(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *tile_count_semaphore_ptr,

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

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int window_size_left,
    int window_size_right,

    int is_causal,
    int is_bf16,
    void *stream
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.tile_count_semaphore = (int*)tile_count_semaphore_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.is_causal = is_causal;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.p_ptr = nullptr; // used for `return_softmax`.

    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    run_mha_fwd(params, (cudaStream_t)stream);
}

extern "C" void run_mha_b(
    void *do_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *dq_ptr,
    void *dk_ptr,
    void *dv_ptr,
    void *dsoftmax_sum,
    void *dq_accum_ptr,
    void *softmax_lse_log2_ptr,
    void *dq_semaphore_ptr,

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

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int window_size_left,
    int window_size_right,

    int is_causal,
    int is_bf16,
    void *stream
) {
    Flash_bwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.do_ptr = do_ptr;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;
    params.dsoftmax_sum = dsoftmax_sum;
    params.dq_accum_ptr = dq_accum_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.softmax_lse_log2_ptr = softmax_lse_log2_ptr;
    params.dq_semaphore = (int*)dq_semaphore_ptr;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.is_causal = is_causal;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.p_ptr = nullptr; // used for `return_softmax`.

    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.do_row_stride = params.o_row_stride;
    params.do_head_stride = params.o_head_stride;
    params.dq_row_stride = params.q_row_stride;
    params.dk_row_stride = params.k_row_stride;
    params.dv_row_stride = params.v_row_stride;
    params.dq_head_stride = params.q_head_stride;
    params.dk_head_stride = params.k_head_stride;
    params.dv_head_stride = params.v_head_stride;

    params.do_batch_stride = params.o_batch_stride;
    params.dq_batch_stride = params.q_batch_stride;
    params.dk_batch_stride = params.k_batch_stride;
    params.dv_batch_stride = params.v_batch_stride;

    run_mha_bwd(params, (cudaStream_t)stream);
}

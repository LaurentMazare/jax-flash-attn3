#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"
#include "flash_bwd_launch_template.h"

namespace gpu_ops {

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

void set_params(Flash_fwd_params &params, const MHAParams &d) {
  // All stride are in elements, not bytes.
  params.q_batch_stride = d.q_batch_stride;
  params.k_batch_stride = d.k_batch_stride;
  params.v_batch_stride = d.v_batch_stride;
  params.o_batch_stride = d.o_batch_stride;

  params.q_row_stride = d.q_row_stride;
  params.k_row_stride = d.k_row_stride;
  params.v_row_stride = d.v_row_stride;
  params.o_row_stride = d.o_row_stride;
  params.q_head_stride = d.q_head_stride;
  params.k_head_stride = d.k_head_stride;
  params.v_head_stride = d.v_head_stride;
  params.o_head_stride = d.o_head_stride;

  // Set the dimensions.
  params.b = d.b;
  params.h = d.h;
  params.h_k = d.h_k;
  params.h_h_k_ratio = d.h / d.h_k;
  params.seqlen_q = d.seqlen_q;
  params.seqlen_k = d.seqlen_k;
  params.seqlen_q_rounded = d.seqlen_q_rounded;
  params.seqlen_k_rounded = d.seqlen_k_rounded;
  params.d = d.d;
  params.d_rounded = d.d_rounded;
  params.is_causal = d.is_causal;

  // // https://github.com/Dao-AILab/flash-attention/blob/72e27c6320555a37a83338178caa25a388e46121/csrc/flash_attn/flash_api.cpp#L107
  // if (d.softcap > 0.0) {
  //     params.softcap = d.softmax_scale / d.softcap;
  //     params.scale_softmax = d.softcap;
  //     params.scale_softmax_log2 = d.softcap * M_LOG2E;
  // } else {
  //     params.softcap = 0.0;
  //     params.scale_softmax = d.softmax_scale;
  //     params.scale_softmax_log2 = d.softmax_scale * M_LOG2E;
  // }
  params.scale_softmax = d.softmax_scale;
  params.scale_softmax_log2 = d.softmax_scale * M_LOG2E;

  params.p_dropout = 1.; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_bf16 = d.is_bf16;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.p_ptr = nullptr; // used for `return_softmax`.

  params.window_size_left = d.window_size_left;
  params.window_size_right = d.is_causal ? 0 : d.window_size_right;
}

void run_mha_fwd_j(cudaStream_t stream, void **buffers,
                   const char *opaque,
                   std::size_t opaque_len) {
  const MHAParams &d = *UnpackDescriptor<MHAParams>(opaque, opaque_len);
  Flash_fwd_params params;
  // Reset the parameters
  memset(&params, 0, sizeof(params));
  set_params(params, d);

  params.q_ptr = buffers[0];
  params.k_ptr = buffers[1];
  params.v_ptr = buffers[2];
  params.tile_count_semaphore = (int*)buffers[3];
  params.o_ptr = buffers[4];
  params.softmax_lse_ptr = buffers[5];

  run_mha_fwd(params, stream);
}

void run_mha_bwd_j(cudaStream_t stream, void **buffers,
                   const char *opaque,
                   std::size_t opaque_len) {
  // The order of the buffers is specified in the jax integration layer.
  // grad, output, softmax_lse, q, k, v, dq, dk, dv

  const MHAParams &d = *UnpackDescriptor<MHAParams>(opaque, opaque_len);
  Flash_bwd_params params;
  // Reset the parameters, we treat params as a Flash_fwd_params here as Flash_bwd params inherit
  // from Flash_bwd_params. The same thing is done in the PyTorch integration.
  // https://github.com/Dao-AILab/flash-attention/blob/3566596ad867ee415dd3c12616dd50c610176f6c/csrc/flash_attn/flash_api.cpp#L153
  memset(&params, 0, sizeof(params));
  set_params(params, d);

  params.do_ptr = buffers[0];
  params.o_ptr = buffers[1];
  params.softmax_lse_ptr = buffers[2];
  params.q_ptr = buffers[3];
  params.k_ptr = buffers[4];
  params.v_ptr = buffers[5];
  params.dq_ptr = buffers[6];
  params.dk_ptr = buffers[7];
  params.dv_ptr = buffers[8];
  params.dsoftmax_sum = buffers[9];
  params.dq_accum_ptr = buffers[10];
  params.softmax_lse_log2_ptr = buffers[11];
  params.dq_semaphore = (int*)buffers[12];
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;

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

  run_mha_bwd(params, stream);
}
}

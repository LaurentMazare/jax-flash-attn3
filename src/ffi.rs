use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha_f(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        tile_count_sempahore_ptr: *const c_void,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        window_size_left: c_int,
        window_size_right: c_int,

        is_causal: c_int,
        is_bf16: c_int,
        stream: *const c_void,
    );

    pub(crate) fn run_mha_b(
        do_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        dq_ptr: *const c_void,
        dk_ptr: *const c_void,
        dv_ptr: *const c_void,
        dsoftmax_sum: *const c_void,
        dq_accum_ptr: *const c_void,
        softmax_lse_log2_ptr: *const c_void,
        dq_semaphore_ptr: *const c_void,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        window_size_left: c_int,
        window_size_right: c_int,

        is_causal: c_int,
        is_bf16: c_int,
        stream: *const c_void,
    );
}

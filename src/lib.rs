use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyCapsule};

mod ffi;

fn w<E: std::error::Error>(err: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Func {
    MhaFwd,
    MhaBwd,
}

#[pyclass]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Params {
    func: Func,
    batch_size: usize,
    seqlen_q: usize,
    seqlen_q_rounded: usize,
    seqlen_k: usize,
    seqlen_k_rounded: usize,
    num_heads: usize,
    num_heads_k: usize,
    head_size: usize,
    head_size_rounded: usize,
    softmax_scale: f32,
    window_size_left: i32,
    window_size_right: i32,
    is_causal: bool,
    is_bf16: bool,
}

#[pymethods]
impl Params {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        batch_size: usize,
        seqlen_q: usize,
        seqlen_q_rounded: usize,
        seqlen_k: usize,
        seqlen_k_rounded: usize,
        num_heads: usize,
        num_heads_k: usize,
        head_size: usize,
        head_size_rounded: usize,
        softmax_scale: f32,
        window_size_left: i32,
        window_size_right: i32,
        is_causal: bool,
        is_bf16: bool,
        is_bwd: bool,
    ) -> Self {
        let func = if is_bwd { Func::MhaBwd } else { Func::MhaFwd };
        Self {
            batch_size,
            seqlen_q,
            seqlen_q_rounded,
            seqlen_k,
            seqlen_k_rounded,
            num_heads,
            num_heads_k,
            head_size,
            head_size_rounded,
            softmax_scale,
            window_size_left,
            window_size_right,
            is_causal,
            is_bf16,
            func,
        }
    }

    fn serialize(&self) -> PyResult<PyObject> {
        let data = bincode::serialize(self).map_err(w)?;
        let bytes: PyObject = Python::with_gil(|py| PyBytes::new_bound(py, &data).into());
        Ok(bytes)
    }
}

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn rust_fn(
    stream: *const std::ffi::c_void,
    buffers: *const *const std::ffi::c_void,
    opaque: *const std::primitive::char,
    opaque_len: usize,
) {
    let opaque = std::slice::from_raw_parts(opaque as *const u8, opaque_len);
    let p: Params = match bincode::deserialize(opaque) {
        Ok(params) => params,
        Err(_) => {
            eprintln!("error unwrapping params");
            return;
        }
    };
    let head_size = p.head_size as u32;
    let seqlen_q = p.seqlen_q as u32;
    let seqlen_k = p.seqlen_k as u32;
    let num_heads = p.num_heads as u32;
    let num_heads_k = p.num_heads_k as u32;
    match p.func {
        Func::MhaFwd => ffi::run_mha_f(
            /* q_ptr */ *buffers.add(0),
            /* k_ptr */ *buffers.add(1),
            /* v_ptr */ *buffers.add(2),
            /* o_ptr */ *buffers.add(4),
            /* softmax_lse_ptr */ *buffers.add(5),
            /* tile_count_semaphore is buffer 3 as it's an input */
            *buffers.add(3),
            /* q_batch_stride */ seqlen_q * num_heads * head_size,
            /* k_batch_stride */ seqlen_k * num_heads_k * head_size,
            /* v_batch_stride */ seqlen_k * num_heads_k * head_size,
            /* o_batch_stride */ seqlen_q * num_heads * head_size,
            /* q_row_stride */ num_heads * head_size,
            /* k_row_stride */ num_heads_k * head_size,
            /* v_row_stride */ num_heads_k * head_size,
            /* o_row_stride */ num_heads * head_size,
            /* q_head_stride */ head_size,
            /* k_head_stride */ head_size,
            /* v_head_stride */ head_size,
            /* o_head_stride */ head_size,
            /* b */ p.batch_size as u32,
            /* h */ num_heads,
            /* h_k */ num_heads_k,
            /* d */ head_size,
            /* d_rounded */ p.head_size_rounded as u32,
            /* softmax_scale */ p.softmax_scale,
            /* seqlen_q */ seqlen_q,
            /* seqlen_k */ seqlen_k,
            /* seqlen_q_rounded */ p.seqlen_q_rounded as u32,
            /* seqlen_k_rounded */ p.seqlen_k_rounded as u32,
            /* window_size_left */ p.window_size_left as i32,
            /* window_size_right */ p.window_size_right as i32,
            /* is_causal */ p.is_causal as i32,
            /* is_bf16 */ p.is_bf16 as i32,
            /* stream */ stream,
        ),
        Func::MhaBwd => ffi::run_mha_b(
            /* do_ptr */ *buffers.add(0),
            /* o_ptr */ *buffers.add(1),
            /* softmax_lse_ptr */ *buffers.add(2),
            /* q_ptr */ *buffers.add(3),
            /* k_ptr */ *buffers.add(4),
            /* v_ptr */ *buffers.add(5),
            /* dq_ptr */ *buffers.add(6),
            /* dk_ptr */ *buffers.add(7),
            /* dv_ptr */ *buffers.add(8),
            /* dsoftmax_sum */ *buffers.add(9),
            /* dq_accum_ptr */ *buffers.add(10),
            /* softmax_lse_log2_ptr */ *buffers.add(11),
            /* dq_semaphore_ptr */ *buffers.add(12),
            /* q_batch_stride */ seqlen_q * num_heads * head_size,
            /* k_batch_stride */ seqlen_k * num_heads_k * head_size,
            /* v_batch_stride */ seqlen_k * num_heads_k * head_size,
            /* o_batch_stride */ seqlen_q * num_heads * head_size,
            /* q_row_stride */ num_heads * head_size,
            /* k_row_stride */ num_heads_k * head_size,
            /* v_row_stride */ num_heads_k * head_size,
            /* o_row_stride */ num_heads * head_size,
            /* q_head_stride */ head_size,
            /* k_head_stride */ head_size,
            /* v_head_stride */ head_size,
            /* o_head_stride */ head_size,
            /* b */ p.batch_size as u32,
            /* h */ num_heads,
            /* h_k */ num_heads_k,
            /* d */ head_size,
            /* d_rounded */ p.head_size_rounded as u32,
            /* softmax_scale */ p.softmax_scale,
            /* seqlen_q */ seqlen_q,
            /* seqlen_k */ seqlen_k,
            /* seqlen_q_rounded */ p.seqlen_q_rounded as u32,
            /* seqlen_k_rounded */ p.seqlen_k_rounded as u32,
            /* window_size_left */ p.window_size_left as i32,
            /* window_size_right */ p.window_size_right as i32,
            /* is_causal */ p.is_causal as i32,
            /* is_bf16 */ p.is_bf16 as i32,
            /* stream */ stream,
        ),
    }
}

extern "C" {
    fn cpp_wrap_ptr() -> *const std::ffi::c_void;
}

static XLA_NAME: &[u8] = b"xla._CUSTOM_CALL_TARGET\0";

#[pyfunction]
fn xla_registrations() -> PyResult<PyObject> {
    let cpp_wrap_ptr = unsafe { cpp_wrap_ptr() };
    let dict = Python::with_gil(|py| -> PyResult<PyObject> {
        let rust_fn = unsafe {
            pyo3::ffi::PyCapsule_New(
                cpp_wrap_ptr as *mut std::ffi::c_void,
                XLA_NAME.as_ptr() as *const i8,
                None,
            )
        };
        let rust_fn: Py<PyCapsule> = unsafe { Py::from_owned_ptr_or_err(py, rust_fn)? };
        let dict = IntoPyDict::into_py_dict_bound([("rust_fn", rust_fn)], py);
        Ok(dict.into())
    })?;
    Ok(dict)
}

#[pymodule]
#[pyo3(name = "_jflash_attn")]
fn jflash_attn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xla_registrations, m)?)?;
    m.add_class::<Params>()?;
    Ok(())
}

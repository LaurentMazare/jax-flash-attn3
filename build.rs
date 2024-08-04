// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

const KERNEL_FILES: [&str; 16] = [
    "flash_api_rust.cu",
    "flash_bwd_hdim128_bf16_sm90.cu",
    "flash_bwd_hdim128_fp16_sm90.cu",
    "flash_bwd_hdim64_bf16_sm90.cu",
    "flash_bwd_hdim64_fp16_sm90.cu",
    "flash_bwd_hdim96_bf16_sm90.cu",
    "flash_bwd_hdim96_fp16_sm90.cu",
    "flash_fwd_hdim128_bf16_sm90.cu",
    "flash_fwd_hdim128_e4m3_sm90.cu",
    "flash_fwd_hdim128_fp16_sm90.cu",
    "flash_fwd_hdim256_bf16_sm90.cu",
    "flash_fwd_hdim256_e4m3_sm90.cu",
    "flash_fwd_hdim256_fp16_sm90.cu",
    "flash_fwd_hdim64_bf16_sm90.cu",
    "flash_fwd_hdim64_e4m3_sm90.cu",
    "flash_fwd_hdim64_fp16_sm90.cu",
];

fn main() -> Result<()> {
    pyo3_build_config::add_extension_module_link_args();

    let num_cpus = std::env::var("RAYON_NUM_THREADS")
        .map_or_else(|_| num_cpus::get_physical(), |s| usize::from_str(&s).unwrap());

    rayon::ThreadPoolBuilder::new().num_threads(num_cpus).build_global().unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed=csrc/{kernel_file}");
    }
    println!("cargo:rerun-if-changed=csrc/flash_fwd_kernel.h");
    println!("cargo:rerun-if-changed=csrc/flash_fwd_launch_template.h");
    println!("cargo:rerun-if-changed=csrc/flash.h");
    println!("cargo:rerun-if-changed=csrc/philox.cuh");
    println!("cargo:rerun-if-changed=csrc/softmax.h");
    println!("cargo:rerun-if-changed=csrc/utils.h");
    println!("cargo:rerun-if-changed=csrc/kernel_traits.h");
    println!("cargo:rerun-if-changed=csrc/block_info.h");
    println!("cargo:rerun-if-changed=csrc/static_switch.h");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            match path.canonicalize() {
                Ok(path) => path,
                Err(err) => anyhow::bail!(
                    "missing directory: {path:?}, {err:?} (current dir is {:?})",
                    std::env::current_dir()?
                ),
            }
        }
    };
    let cuda_root = set_cuda_include_dir()?;

    let ccbin_env = std::env::var("NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    let out_file = build_dir.join("libflashattention.a");

    let kernel_dir = PathBuf::from("csrc");
    let cu_files: Vec<_> = KERNEL_FILES
        .iter()
        .map(|f| {
            let mut obj_file = out_dir.join(f);
            obj_file.set_extension("o");
            (kernel_dir.join(f), obj_file)
        })
        .collect();
    let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
    let should_compile = if out_file.exists() {
        kernel_dir.read_dir().expect("csrc folder should exist").any(|entry| {
            if let (Ok(entry), Ok(out_modified)) = (entry, &out_modified) {
                let in_modified = entry.metadata().unwrap().modified().unwrap();
                in_modified.duration_since(*out_modified).is_ok()
            } else {
                true
            }
        })
    } else {
        true
    };
    if should_compile {
        cu_files
            .par_iter()
            .map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                command
                    .arg("-std=c++17")
                    .arg("-O3")
                    .args(["-gencode", "arch=compute_90a,code=sm_90a"])
                    .arg("-Xcompiler=-fPIC")
                    .arg("-U__CUDA_NO_HALF_OPERATORS__")
                    .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                    .arg("-U__CUDA_NO_HALF2_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                    .arg("-DNDEBUG")
                    .arg("-DQBLKSIZE=128")
                    .arg("-DKBLKSIZE=128")
                    .arg("-DCTA256")
                    .arg("-DDQINRMEM")
                    .arg("-c")
                    .args(["-o", obj_file.to_str().unwrap()])
                    .args(["--default-stream", "per-thread"])
                    .arg("-Icutlass/include")
                    .arg(format!("-I{cuda_root:?}/include"))
                    .arg("--expt-relaxed-constexpr")
                    .arg("--expt-extended-lambda")
                    .arg("--use_fast_math")
                    .arg("--verbose");
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .context("failed spawning nvcc")?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<()>>()?;
        let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
        let mut command = std::process::Command::new("nvcc");
        command.arg("--lib").args(["-o", out_file.to_str().unwrap()]).args(obj_files);
        let output = command.spawn().context("failed spawning nvcc")?.wait_with_output()?;
        if !output.status.success() {
            anyhow::bail!(
                "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                &command,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        }
    }
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=static=cudart_static");
    // println!("cargo:rustc-link-lib=static=cublasLt_static");
    // println!("cargo:rustc-link-lib=static=cublas_static");
    // println!("cargo:rustc-link-lib=static=nvrtc_static");
    println!("cargo:rustc-link-lib=static=nvptxcompiler_static");
    // println!("cargo:rustc-link-lib=static=nvrtc-builtins_static");

    /* laurent: I tried using the cc cuda integration as below but this lead to ptaxs never
       finishing to run for some reason. Calling nvcc manually worked fine.
    cc::Build::new()
        .cuda(true)
        .include("cutlass/include")
        .flag("--expt-relaxed-constexpr")
        .flag("--default-stream")
        .flag("per-thread")
        .flag(&format!("--gpu-architecture=sm_{compute_cap}"))
        .file("csrc/flash_fwd_hdim32_fp16_sm80.cu")
        .compile("flashattn");
    */
    Ok(())
}

fn set_cuda_include_dir() -> Result<PathBuf> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDNN_LIB"];
    let env_vars =
        env_vars.into_iter().map(std::env::var).filter_map(Result::ok).map(Into::<PathBuf>::into);
    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::<PathBuf>::into);
    let root = env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .context("cannot find include/cuda.h")?;
    println!("cargo:rustc-env=CUDA_INCLUDE_DIR={}", root.join("include").display());
    println!("cargo:rustc-link-search=native={}", root.join("lib64").display());
    Ok(root)
}

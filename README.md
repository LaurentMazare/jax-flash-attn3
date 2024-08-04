# jax-flash-attn

> [!WARNING]  
> These bindings are very experimental. If you use them, double check that the
> outputs are reasonable. The current tests only verify this for the most simple
> setups.
>
> Only a subset of the options are a supported, in particular GQA/MQA haven't
> been tested.

This repo contains bindings for [FlashAttention3](https://github.com/Dao-AILab/flash-attention)
in JAX. There are two versions for these bindings, a C++ version
`jax_flash_attn` and a Rust version `jflash_attn`.

The BSD-3 license that holds for the flash-attention repo also applies here.

## Building the C++ Version

Build a wheel file. `-j32` will compile 32 cuda kernels in parallel which could exhaust memory on boxes with
less than 100GB.
```bash
python setup.py bdist_wheel -- -- -j32
```

Build locally for development.
```bash
python setup.py build_ext -i -- -- -j32
python test.py # run some tests and benchmarks
```

This may require you to install the two following pip packages:
```bash
pip install scikit_build
pip install "pybind11[global]"
```

## Building the Rust Version

In order to build a python package as a wheel, run `maturin build --release`.
In order to build a python package and install it in the current virtual
enviroment, run `maturin develop`.

## Running the Tests and Benchmarks

First compile the C++ and/or Rust package and install them locally. Use the
following to run the tests.
```bash
python test.py --bindings cpp
python test.py --bindings rust
```

And use the `--bench` flag to run the benchmarks instead of the tests.

```bash
python test.py --bindings cpp --bench True
python test.py --bindings rust --bench True
```

## Benchmarks (H100 80G HBM3)

This measures the time spent in the attention layer for three different implementations.
- `flash-attn`: uses the optimized flash-attention kernel. 
- `attn-einsum`: uses a simple attention implementation based on einsum.
- `attn-flax`: uses `flax.linen.dot_product_attention`.
Timings include the forward pass only for the first lines and both the forward
and backward passes for the lines that start with `bwd`. The second column is the
sequence length (the batch size is adapted so as to have a reasonable amount of
computation).

```
flash-attn           512  0.96ms     71.6 TFLOPS (std 0.39ms, min 0.79ms, max 2.43ms)
attn-flax            512  1.90ms     36.1 TFLOPS (std 0.44ms, min 1.64ms, max 3.46ms)
flash-attn          1024  1.04ms    131.8 TFLOPS (std 0.25ms, min 0.88ms, max 1.74ms)
attn-flax           1024  1.13ms    122.0 TFLOPS (std 0.27ms, min 0.98ms, max 1.94ms)
flash-attn          2048  1.16ms    237.6 TFLOPS (std 0.13ms, min 1.08ms, max 1.58ms)
attn-flax           2048  1.44ms    191.2 TFLOPS (std 0.39ms, min 1.25ms, max 2.68ms)
flash-attn          4096  1.59ms    346.2 TFLOPS (std 0.30ms, min 1.45ms, max 2.82ms)
attn-flax           4096  1.91ms    287.8 TFLOPS (std 0.33ms, min 1.75ms, max 3.20ms)
flash-attn          8192  2.27ms    483.9 TFLOPS (std 0.18ms, min 2.16ms, max 3.05ms)
attn-flax           8192  2.97ms    370.4 TFLOPS (std 0.36ms, min 2.79ms, max 4.17ms)
flash-attn         16384  3.88ms    566.6 TFLOPS (std 0.29ms, min 3.71ms, max 4.67ms)
attn-flax          16384 22.14ms     99.3 TFLOPS (std 0.56ms, min 21.54ms, max 23.44ms)
bwd flash-attn       512  2.23ms    107.9 TFLOPS (std 0.30ms, min 2.04ms, max 2.93ms)
bwd attn-flax        512  3.30ms     72.9 TFLOPS (std 0.17ms, min 3.17ms, max 3.84ms)
bwd flash-attn      1024  2.54ms    189.4 TFLOPS (std 0.31ms, min 2.29ms, max 3.28ms)
bwd attn-flax       1024  4.79ms    100.4 TFLOPS (std 0.38ms, min 4.60ms, max 5.92ms)
bwd flash-attn      2048  3.29ms    292.1 TFLOPS (std 0.50ms, min 2.89ms, max 4.42ms)
bwd attn-flax       2048  7.66ms    125.5 TFLOPS (std 0.35ms, min 7.48ms, max 8.52ms)
bwd flash-attn      4096  4.25ms    452.7 TFLOPS (std 0.34ms, min 4.03ms, max 5.20ms)
bwd attn-flax       4096 13.70ms    140.4 TFLOPS (std 0.51ms, min 13.17ms, max 15.23ms)
bwd flash-attn      8192  7.86ms    489.7 TFLOPS (std 1.57ms, min 7.02ms, max 13.35ms)
bwd attn-flax       8192 25.31ms    152.0 TFLOPS (std 0.51ms, min 24.80ms, max 26.60ms)
bwd flash-attn     16384 13.62ms    565.3 TFLOPS (std 0.49ms, min 13.09ms, max 15.08ms)
bwd attn-flax      16384 47.84ms    160.9 TFLOPS (std 0.44ms, min 47.54ms, max 49.61ms)
```

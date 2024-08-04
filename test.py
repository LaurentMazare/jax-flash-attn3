import argparse
import time

import sys
import numpy as np
import jax
import jax.numpy as jnp
import flax

parser = argparse.ArgumentParser()
parser.add_argument("--bindings", type=str, default="cpp")
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--bench", type=bool, default=False)
args = parser.parse_args()

if args.bindings == "cpp":
    import jax_flash_attn

    print(jax_flash_attn.__file__)
    from jax_flash_attn import run_mha
elif args.bindings == "rust":
    import jflash_attn

    print(jflash_attn.__file__)
    from jflash_attn import run_mha
else:
    raise ValueError('unsupported bindings "{args.bindings}", use "cpp" or "rust"')


def attn_einsum(q, k, v, mask=None):
    softmax_scale = q.shape[-1] ** -0.5
    qk = jnp.einsum("bqhd,bkhd->bhqk", q, k)

    if mask is not None:
        qk = qk + jnp.log(mask)
    attn_weights = jax.nn.softmax(qk * softmax_scale, axis=-1)
    attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    return attn


def test_fwd(qkv_shape, max_err, is_causal, dtype=jnp.float16):
    _b_size, seqlen, _num_heads, head_dim = qkv_shape
    rng_q = jax.random.PRNGKey(0)
    q = jax.random.normal(rng_q, qkv_shape, dtype=dtype)
    rng_k = jax.random.PRNGKey(1)
    k = jax.random.normal(rng_k, qkv_shape, dtype=dtype)
    rng_v = jax.random.PRNGKey(2)
    v = jax.random.normal(rng_v, qkv_shape, dtype=dtype) / seqlen

    mask = None
    if is_causal:
        mask = jnp.tril(jnp.ones((seqlen, seqlen)))

    softmax_scale = head_dim**-0.5
    attn_mha = run_mha(q, k, v, is_causal=is_causal, softmax_scale=softmax_scale)
    attn_ein = attn_einsum(q, k, v, mask=mask)
    attn_flax = flax.linen.dot_product_attention(q, k, v, mask=mask)

    diff_mha_ein = (attn_mha - attn_ein).max()
    diff_mha_flax = (attn_mha - attn_flax).max()
    diff_ein_flax = (attn_ein - attn_flax).max()
    if args.verbose:
        print("fwd", diff_mha_ein, diff_mha_flax, diff_ein_flax)
    if not (diff_mha_ein <= max_err):  # be cautious about handling nans
        print(
            "FAIL    fwd",
            qkv_shape,
            diff_mha_ein,
            diff_mha_flax,
            diff_ein_flax,
            is_causal,
            dtype,
        )


def test_bwd(qkv_shape, max_err, is_causal, dtype=jnp.float16):
    _b_size, seqlen, _num_heads, head_dim = qkv_shape
    rng_q = jax.random.PRNGKey(0)
    q = jax.random.normal(rng_q, qkv_shape, dtype=dtype)
    rng_k = jax.random.PRNGKey(1)
    k = jax.random.normal(rng_k, qkv_shape, dtype=dtype)
    rng_v = jax.random.PRNGKey(2)
    v = jax.random.normal(rng_v, qkv_shape, dtype=dtype)

    mask = None
    if is_causal:
        mask = jnp.tril(jnp.ones((seqlen, seqlen)))

    def loss_mha(q, k, v):
        softmax_scale = head_dim**-0.5
        predictions = run_mha(q, k, v, is_causal=is_causal, softmax_scale=softmax_scale)
        return jnp.sum(predictions)

    loss_mha_grad = jax.grad(loss_mha, (0, 1, 2))

    def loss_flax(q, k, v):
        predictions = flax.linen.dot_product_attention(q, k, v, mask=mask)
        return jnp.sum(predictions)

    loss_flax_grad = jax.grad(loss_flax, (0, 1, 2))

    dq_mha, dk_mha, dv_mha = loss_mha_grad(q, k, v)
    dq_flax, dk_flax, dv_flax = loss_flax_grad(q, k, v)

    dq_diff = ((dq_mha - dq_flax) ** 2).mean()
    dk_diff = ((dk_mha - dk_flax) ** 2).mean()
    dv_diff = ((dv_mha - dv_flax) ** 2).mean()
    if args.verbose:
        print("bwd", dq_diff, dk_diff, dv_diff)
    if not (
        dq_diff <= max_err and dk_diff <= max_err and dv_diff <= max_err
    ):  # be cautious about nans.
        print(dq_mha[0, 1, 0])
        print(dq_flax[0, 1, 0])
        print(dq_mha[0, 1, 0] / dq_flax[0, 1, 0])
        print("FAIL    bwd", qkv_shape, dq_diff, dk_diff, dv_diff, is_causal, dtype)


TEST_CASES = [
    ((1, 20, 16, 32), 1e-3),
    ((1, 2, 1, 64), 5e-4),
    ((16, 100, 28, 64), 2e-4),
    ((16, 512, 32, 128), 1e-4),
    ((21, 50, 17, 160), 5e-4),
]

if not args.bench:
    for _qkv, _max_err in TEST_CASES:
        for _dtype in [jnp.float16, jnp.bfloat16]:
            test_fwd(_qkv, _max_err, is_causal=False, dtype=_dtype)
            test_fwd(_qkv, _max_err, is_causal=True, dtype=_dtype)
            if _qkv[-1] in (64, 128):
                # TODO: this is currently broken and the differences in dq seems to
                # be some constant multiplier that depends on the sequence length.
                test_bwd(_qkv, _max_err, is_causal=False, dtype=_dtype)
                # TODO: test the causal bwd.
                test_bwd(_qkv, _max_err, is_causal=True, dtype=_dtype)


def bench(label, fwd, b_sz, seq_len, n_heads, dim, n_run=20, n_warmup=4, bwd=False):
    # the flops below only include the matmul of the forward pass
    flops = 4 * b_sz * seq_len * seq_len * n_heads * dim  # b.q.k.h.d
    if bwd:
        flops *= 3.5
    if bwd:

        def loss(q, k, v):
            return jnp.sum(fwd(q, k, v))

        f = jax.grad(loss, (0, 1, 2))
    else:
        f = fwd
    qkv_shape = b_sz, seq_len, n_heads, dim

    def normal(seed):
        rng = jax.random.PRNGKey(seed)
        return jax.random.normal(rng, qkv_shape, dtype=jnp.float16)

    dts = []
    for i in range(n_warmup + n_run):
        q = normal(3 * i)
        k = normal(3 * i + 1)
        v = normal(3 * i + 2) / seq_len
        start_time = time.perf_counter()
        res = f(q, k, v)
        if bwd:
            res = res[0]
        res = res.block_until_ready()
        res = float(res.sum())
        dt = time.perf_counter() - start_time
        dts.append(dt)
    # print(dts)
    dts = dts[n_warmup:]
    dts = np.array(dts)
    min_ms = np.min(dts) * 1000
    max_ms = np.max(dts) * 1000
    mean_ms = np.mean(dts) * 1000
    std_ms = np.std(dts) * 1000
    gflops = flops / np.mean(dts) / 1e12
    print(
        f"{label:16} {seq_len:7} {mean_ms:5.2f}ms {gflops:8.1f} TFLOPS (std {std_ms:.2f}ms, min {min_ms:.2f}ms, max {max_ms:.2f}ms)"
    )


if args.bench:
    run_mha_jit = jax.jit(run_mha)
    attn_einsum_jit = jax.jit(attn_einsum)
    attn_flax_jit = jax.jit(flax.linen.dot_product_attention)

    # Values taken from:
    # https://github.com/Dao-AILab/flash-attention/blob/2c3baba4a63c4007c8a132c5380edc9430f88a22/benchmarks/benchmark_flash_attention.py#L74C1-L77C11
    BSIZE_SEQLEN_VALS = [
        (32, 512),
        (16, 1024),
        (8, 2048),
        (4, 4096),
        (2, 8192),
        (1, 16384),
    ]
    HEADDIM = 128
    DIM = 2048
    n_heads = DIM // HEADDIM

    for b_sz, seqlen in BSIZE_SEQLEN_VALS:
        bench("flash-attn ", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM)
        # bench("attn-einsum ", attn_einsum_jit, b_sz, seqlen, n_heads, HEADDIM)
        bench("attn-flax ", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM)

    for b_sz, seqlen in BSIZE_SEQLEN_VALS:
        bench("bwd flash-attn ", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)
        # bench("bwd attn-einsum", attn_einsum_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)
        bench("bwd attn-flax", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)

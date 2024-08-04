from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="jax_flash_attn",
    version="0.0.1",
    description="Flash attention CUDA kernels for jax",
    author="",
    packages=["jax_flash_attn"],
    cmake_install_dir="jax_flash_attn",
    python_requires=">=3.10",
)

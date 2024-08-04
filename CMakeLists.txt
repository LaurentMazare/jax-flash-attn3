# Adapted from https://github.com/dfm/extending-jax/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.15...3.26)
project(jax_flash_attn LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/csrc)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/cutlass/include)
  file(GLOB KERNEL_FILES ${CMAKE_CURRENT_LIST_DIR}/csrc/flash_*wd_hdim*.cu)
  message(KF="${KERNEL_FILES}")
  pybind11_add_module(
    _jax_flash_attn
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/csrc/flash_attn_ops.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csrc/flash_api.cu)
  target_compile_options(_jax_flash_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -gencode arch=compute_90a,code=sm_90a;
      --expt-relaxed-constexpr;
      -DEXECMODE=0;
      -std=c++17;
      -U__CUDA_NO_HALF_OPERATORS__;
      -U__CUDA_NO_HALF_CONVERSIONS__;
      -U__CUDA_NO_BFLOAT16_OPERATORS__;
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__;
      -U__CUDA_NO_BFLOAT162_OPERATORS__;
      -U__CUDA_NO_BFLOAT162_CONVERSIONS__;
      --expt-relaxed-constexpr;
      --expt-extended-lambda;
      --use_fast_math;
      --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage;  # printing out number of registers
      -lineinfo;
      -DCUTLASS_DEBUG_TRACE_LEVEL=0;  # Can toggle for debugging
      -DNDEBUG;  # Important, otherwise performance is severely impacted
      -DQBLKSIZE=128;
      -DKBLKSIZE=128;
      -DCTA256;
      -DDQINRMEM;
      >) 
  target_link_libraries(_jax_flash_attn PRIVATE ${CUDA_LIBRARIES} -lcuda)
  set_property(TARGET _jax_flash_attn PROPERTY CUDA_ARCHITECTURES 90a)
  install(TARGETS _jax_flash_attn LIBRARY DESTINATION .)

  add_library(
    cc_jax_flash_attn
    STATIC
    ${KERNEL_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/csrc/flash_api.cu)
  target_compile_options(cc_jax_flash_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -gencode arch=compute_90a,code=sm_90a;
      --expt-relaxed-constexpr;
      -DEXECMODE=0;
      -std=c++17;
      -U__CUDA_NO_HALF_OPERATORS__;
      -U__CUDA_NO_HALF_CONVERSIONS__;
      -U__CUDA_NO_BFLOAT16_OPERATORS__;
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__;
      -U__CUDA_NO_BFLOAT162_OPERATORS__;
      -U__CUDA_NO_BFLOAT162_CONVERSIONS__;
      --expt-relaxed-constexpr;
      --expt-extended-lambda;
      --use_fast_math;
      --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage;  # printing out number of registers
      -lineinfo;
      -DCUTLASS_DEBUG_TRACE_LEVEL=0;  # Can toggle for debugging
      -DNDEBUG;  # Important, otherwise performance is severely impacted
      -DQBLKSIZE=128;
      -DKBLKSIZE=128;
      -DCTA256;
      -DDQINRMEM;
      >) 
  set_property(TARGET cc_jax_flash_attn PROPERTY CUDA_ARCHITECTURES 90a)
  install(TARGETS cc_jax_flash_attn LIBRARY DESTINATION .)
else()
  message(FATAL_ERROR "Cannot be built without CUDA")
endif()

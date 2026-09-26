# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project Overview

**TinyTorch** is a minimalistic, educational auto-differentiation / backpropagation
framework written in C++17. It mimics the PyTorch/libTorch API surface (`tinytorch::Tensor`,
`backward()`, optimizers, `nn::Module`) to teach how autograd works — it is *not* a
production framework.

- Language: C++17 (set in root `CMakeLists.txt`, do not lower it)
- Build system: CMake (>= 3.15)
- Dependencies: C++ standard library, OpenMP (required), glog-style `CHECK` macros
  (vendored `External/tiny-glog`), optional CUDA toolkit, googletest submodule
  (`External/googletest`, pinned to v1.17.0) for the unit tests
- Namespace: `tinytorch` (aliases `torch` and `at` exist in `torch.h`/`check.h`)
- Library target: `torch` (shared library), sample executable: `tt_sample`
- License: MIT

## Repository Layout

```
CMakeLists.txt              Root build: options, CUDA/OpenMP detection, CUDA arch resolution, subdirs
cmake/select_compute_arch.cmake  CUDA arch selection (copied from saiga): autodetect / All / explicit
External/tiny-glog/         Vendored minimal glog (CHECK/LogMessage), provides glog::glog target
External/glog/              git submodule (google/glog) — UNUSED by the build, leftover
External/googletest/        git submodule (google/googletest @ v1.17.0), used by the tests
src/
  CMakeLists.txt            Builds shared lib `torch` from ALL *.cpp/*.cu under src/
  torch/
    tiny_torch.h            Main public header (includes core + ops + optimizer + module)
    torch.h                 Adds `namespace torch/at = tinytorch;` aliases
    tiny_torch_config.h     TINYTORCH_API visibility macros, TT_HD, TT_DEBUG, exceptions
    tiny_torch_build_config.h.in  Configured at build time -> defines TT_HAS_CUDA
    tiny_torch_cuda.h       CUDA-only includes (guarded by __has_include(cuda_runtime.h))
    core/
      tensor.h/.cpp         Tensor (thin handle over shared_ptr<TensorImpl>)
      tensor_impl.h/.cpp    TensorImpl: storage ptr, offset, sizes, strides, options, AutogradMeta
      tensor_data.h/.cpp    StorageImpl: raw byte buffer + ownership
      tensor_options.h      TensorOptions (device, dtype, requires_grad, pinned)
      tensor_info.h         TensorInfo<T>/TensorInfoCuda<T>: kernel-facing layout struct (sizes/strides/offset math)
      types.h/.cpp          ScalarType, Device, DeviceType, PaddingMode, InterpolationType
      half.h/.cpp           Custom fp16 `Half` type (CPU side)
      graph.h/.cpp          Autograd: Node, FunctionNode<T>, Edge, Context, AccumulateGrad, GradMode, NoGradGuard
      backward.h/.cpp       backward(loss): reverse-mode traversal of the node graph
      check.h               Just includes tiny_torch.h + namespace aliases
      module.h              nn::Module, ModuleHolder<T>, TORCH_MODULE / TORCH_MODULE_IMPL macros
      optimizer.h/.cpp      optim::Adam, optim::SGDOptimizer
      ops/                  USER-FACING autodiff API + FunctionNode definitions
        all.h               UMBRELLA HEADER — includes every ops header
        ops_operators.*     +, -, *, /, comparisons, compound assignment operators
        ops_unary_functions.*   abs, sqrt, log, exp, sin, cos, relu, sigmoid, softplus, softmax, ...
        ops_math_functions.*    sum, mean, min, max, pow, prod, cumsum, clamp, norm, ...
        ops_functions.*     reshape, permute, slice, cat, stack, copy, to, gather, matmul, conv2d, grid_sample, padding_*
        ops_tensor_creation.*   empty, zeros, ones, full, rand, randn, range, from_blob, ...
        ops_impl.h          SELECT_DEVICE(device, func, ...) dispatch macro -> cpu_impl::/cuda_impl::
        ops_impl_shared.h   Shared elementwise forward/backward functors (UnaryOperators/BinaryOperators),
                            broadcasting helpers (CheckOperatorSizeMatchOneDim, max_size, BackwardExpand)
    cpu/                    CPU kernels, namespace cpu_impl
      ops_impl_cpu.h/.cpp   reductions, indexing, matmul, padding, ...
      binary_operators.*    elementwise binary kernels (OpenMP, SIMD)
      unary_operators.*     elementwise unary kernels (OpenMP)
      conv.h/.cpp           conv2d kernel
      grid_sample.h/.cpp    grid_sample kernel
      ops_impl_cpu_helper.h CASE_MACRO / SWITCH_MACRO_* type-dispatch macros
    cuda/                   CUDA kernels (.cu), namespace cuda_impl
      ops_impl_cuda.h/.cu   same impl surface as cpu, dispatch via CUDA_CASE_MACRO
      binary_operators.* / unary_operators.* / reduce_operators.* / reduce_dim_operators.* / ops_indexing.cu
       grid_sample.*  conv.h/.cu (conv2d kernel)  multi_device.* (MultiDeviceTensor: multi-GPU params/grads)
      tt_cuda.h/.cu         DeviceGuard, streams, TT_CHECK_CUDA_ERROR, CUDA_KERNEL_ASSERT
      cached_memory_allocator.*   CUDA caching allocator
      atomic_minmax.h, reduce_helper.h, ops_impl_cuda_helper.h
samples/
  CMakeLists.txt            Builds tt_sample
  main.cpp                  Runs an Adam optimization loop on CPU (and CUDA if TT_HAS_CUDA)
tests/
  CMakeLists.txt            Builds test_tiny_torch_all (links torch + gtest_main), registers
                            the individual gtest cases with CTest via gtest_discover_tests
  test_utils.h              Shared machinery: DeviceTest fixture (CPU + CUDA when TT_HAS_CUDA),
                            make_tensor/leaf helpers, tensor_near/TT_EXPECT_CLOSE, check_grads
                            (central finite-difference gradient checker), TT_INSTANTIATE_DEVICE_TESTS
  test_tensor_creation.cpp  empty/zeros/ones/full/*_like, rand/randn/randint, range, from_blob
  test_operators.cpp        +-*/ , scalar mixes, comparisons, compound assign, broadcasting + grads
  test_unary_functions.cpp  abs/sqrt/square/log/log1p/exp/sin/cos/sign/round/relu/sigmoid/softplus/softmax
  test_math_functions.cpp   sum/mean/min/max/prod/cumsum/cumprod/pow/clamp/norm/std/abs_sum/prod_sum/median
  test_functions.cpp        reshape/permute/transpose/flip/slice/stack/cat/index_*/gather/padding/matmul/
                            conv2d/sort/poisson/grid_sample (device-parameterized where both backends exist)
  test_tensor_methods.cpp   Tensor member API: accessors, views, in-place ops, math/index methods, item
  test_autograd.cpp         grad mode, NoGradGuard/AutoGradMode, leaf props, graph accumulation
  test_optimizer.cpp        Adam + SGDOptimizer convergence and single-step (hand-verified) updates
  test_module.cpp           nn::Module register/buffers/params/zero_grad/to, ModuleHolder + TORCH_MODULE
  test_device.cpp           CPU<->CUDA transfers, cross-device autograd, CPU/CUDA agreement, MultiDeviceTensor
```

The test suite is parameterized per device: every `TT_INSTANTIATE_DEVICE_TESTS(Suite)` fixture runs
once on CPU and once on CUDA (the CUDA variant is auto-skipped when the build has no CUDA or no GPU
is present). Note: `TEST_P` (not `TEST_F`) must be used for these fixtures — gtest only wires
`INSTANTIATE_TEST_SUITE_P` to `TEST_P`-generated classes.

## Architecture

### Tensor model (PyTorch-style 3-level design)

```
Tensor (value-type handle, shared_ptr<TensorImpl>)
  └─ TensorImpl (sizes, strides, storage_offset, options, AutogradMeta{grad, edge, requires_grad, retain_grad})
       └─ StorageImpl (raw byte buffer, optional ownership, TensorOptions)
```

- `Tensor` in `core/tensor.h` is deliberately a thin wrapper; all state lives in `TensorImpl`
  so that views/copies share storage and autograd state.
- `TensorImpl::create(...)` is the only constructor path (private ctors).
- Autograd state (`AutogradMeta`) is allocated lazily via `set_requires_grad(requires_grad, allocate_grad)`.
- `data_ptr<T>()` statically asserts the dtype matches `T` (use `CppTypeToScalarType<T>`).
- Tensors must not cross devices; binary ops `CHECK_EQ(a.device(), b.device())`.

### Autograd graph

- Every autodiff op is a `struct XNode : public autograd::FunctionNode<XNode>` defined in the
  matching `ops_*.cpp` (inside `namespace autograd`), with two **static** methods:
  ```cpp
  static std::vector<Tensor> forward(Context* ctx, <args...>)   // compute + ctx->save_for_backward({inputs...})
  static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
  ```
- The public entry point simply does `return XNode::forward_and_build_graph(args...)[0];`
  (see e.g. `abs` in `ops_unary_functions.cpp`).
- `FunctionNode::forward_and_build_graph` (core/graph.h) creates the node, records input edges
  in `node->next`, records input metadata in `Context::next_meta`, runs the forward under a
  `NoGradGuard`, and sets `Edge{node, output_index}` on each output tensor.
- `Context` (a.k.a. `AutogradContext`) stores: `saved_tensors` (via `save_for_backward`),
  `saved_data` (`std::map<std::string, IValue>` for non-tensor args like doubles or SizeTypes),
  and `next_meta` (input shapes, used to validate gradient shapes in `node_backward`).
- `IValue` is the variant type for non-tensor arguments (bool, double, int64, Tensor, SizeType,
  Device, custom classes). Forward functions taking non-tensors (e.g. `softplus(Tensor, double)`)
  must declare them as `IValue` parameters in the node and convert with `.toDouble()` etc.
- `backward(loss, grad={}, retain_grad=false)` (core/backward.cpp): starts at `loss`'s edge,
  walks the graph in **reverse sequence order** (stack + `std::sort` by `sequence_nr`),
  accumulates incoming grads per `(node, input_nr)` in `grad_map`, and finally
  `AccumulateGrad` nodes add grads into the leaf `TensorImpl`s. The graph is cleared afterwards —
  **the graph is one-shot**; you must re-run forward each iteration to build it again.
- Gradient flow control: `GradMode::is_enabled()` is **thread-local**; `NoGradGuard`/
  `AutoGradMode` are RAII guards. Forward passes run under `NoGradGuard` internally.
- If an input does not require grad, `backward` may return an undefined (`{}`) Tensor for it —
  callers must honor `ctx->requires_grad_for_input(i)`.
- Broadcasting: elementwise binary ops use `CheckOperatorSizeMatchOneDim` (sizes must match or
  be 1) + `max_size` for output shape, and `BackwardExpand` in backward to sum-reduce gradients
  over the broadcast dims.
- In-place gradient helper: `autograd::MakeInplaceGradient(t)` (graph.cpp) — returns the leaf's
  existing grad for in-place accumulation.

### Device dispatch

- User-facing ops never touch kernels directly. They call the `SELECT_DEVICE(device, func, ...)`
  macro (ops_impl.h) which switches on `device.type()` → `cpu_impl::func(...)` or
  `cuda_impl::func(...)` (the CUDA case is compiled out when `TT_HAS_CUDA` is 0).
- **When implementing a new op you must provide BOTH a `cpu_impl::` and a `cuda_impl::`
  function with the exact same signature** (the CUDA one can be a trivial/`CHECK(false)` stub,
  but the symbol must exist for the `#ifdef TT_HAS_CUDA` case).
- Elementwise math functors (forward/backward) are shared between backends and live in
  `ops_impl_shared.h` (`UnaryOperators::`, `BinaryOperators::`), marked `TT_HD`
  (`__host__ __device__` when CUDA).
- Scalar-type dispatch:
  - CPU: `CASE_MACRO(func<T>, kType, ...)` inside `SWITCH_MACRO_ALL/FLOAT/INT`
    (cpu/ops_impl_cpu_helper.h), operating on `TensorInfo<T>` (int64 indexing).
  - CUDA: `CUDA_CASE_MACRO(...)` (cuda/ops_impl_cuda_helper.h), operating on
    `TensorInfoCuda<T>` (int32 indexing by default — `is_32bit_addressable()` is CHECKed;
    `__half` is used for `kHalf` on device, `Half` on host).
- Kernels: CPU uses OpenMP (`#pragma omp parallel for num_threads(get_num_threads())`,
  optionally `#pragma omp simd`); CUDA kernels use `__launch_bounds__(128)`,
  `cuda::DeviceGuard` to pin the device, and `TT_CHECK_CUDA_ERROR` for runtime calls.
- `TensorInfoBase` (core/tensor_info.h) converts linear indices <-> dim indices via strides;
  `MAX_TENSORINFO_DIMS` is 8 and must not be redefined (compile error if already set).

### Optimizers and modules

- `optim::Adam` (constructor takes `std::vector<Tensor>` + `AdamOptions`; methods `step()`,
  `zero_grad()` — lowercase) and `optim::SGDOptimizer` (methods `Step()`, `ZeroGrad()` —
  **uppercase**; pre-existing inconsistency, do not "fix" without asking).
- `nn::Module` holds `parameters_`/`buffers_`/`modules_` maps; `register_parameter` also
  allocates the grad. `ModuleHolder<T>` + `TORCH_MODULE(Name)` macro provide a
  shared_ptr wrapper (PyTorch-style `Linear` pattern).
- `save()`/`load()` (module.h, ops_functions.h) are **stubs that throw** — serialization is
  not implemented.

## Adding a new op (checklist)

1. Declare the user-facing function in the right header under `src/torch/core/ops/`
   (`ops_unary_functions.h`, `ops_math_functions.h`, `ops_operators.h`, or `ops_functions.h`)
   with the `TINYTORCH_API` export macro (GCC builds use `-fvisibility=hidden`, so any new
   public symbol MUST have `TINYTORCH_API`).
2. In the matching `ops_*.cpp`, define `struct XNode : public autograd::FunctionNode<XNode>`
   with static `forward`/`backward` (use `ctx->save_for_backward({...})` for tensors needed in
   backward; use `ctx->saved_data["name"] = IValue(...)` for scalars/SizeTypes), then the
   one-line public function calling `forward_and_build_graph`.
3. Implement `cpu_impl::x_impl(...)` in `src/torch/cpu/` (OpenMP loop over `TensorInfo<T>`,
   type dispatch via `SWITCH_MACRO_*`) and `cuda_impl::x_impl(...)` in `src/torch/cuda/`
   (kernels over `TensorInfoCuda<T>`, `CUDA_CASE_MACRO`).
4. For elementwise ops, reuse the functor pattern: put the math functor in
   `ops_impl_shared.h` (`UnaryOperators`/`BinaryOperators`) and call
   `cpu_impl`/`cuda_impl` generic kernels (`unary_operator_kernel`, `element_wise_operator`)
   instead of writing a new loop.
5. Backward must return exactly as many gradients as `forward` had tensor/IValue inputs
   (`num_inputs_of_forward`), and each defined gradient must match the input shape.
 6. Add a googletest case for the op in the matching `tests/test_*.cpp` (see the layout above).
    For differentiable ops, prefer `tttest::check_grads({leaves}, loss_fn)` which verifies the
    analytic gradient against central finite differences; otherwise build leaf tensors, run
    `tinytorch::backward(loss)`, and `EXPECT_*` the resulting `.grad()` values.
    Device-dependent expectations: use the `TT_INSTANTIATE_DEVICE_TESTS(Foo)` + `TEST_P(Foo, ...)`
    pattern (the macro declares the fixture and registers CPU/CUDA runs). If a test exposes a
    genuine library bug, do NOT fix the library — mark the test with `GTEST_SKIP()` + a
    `// KNOWN BUG:` comment explaining the defect and where it lives.

## Build (Linux)

### Prerequisites

- CMake >= 3.15
- GCC (tested with 13) or Clang, C++17
- OpenMP (`libgomp` / `libomp`) — **required**
- x86-64 CPU: the build hard-codes `-msse4.1 -mavx -mavx2 -mfma` for the GNU frontend
- Optional: NVIDIA CUDA Toolkit >= 11.8 (without it, the build automatically falls back to
  CPU-only, which is fully functional)

The build uses the **vendored** `External/tiny-glog` (checked into the repo) for the glog
macros. For the tests, CMake uses the `External/googletest` submodule — a fresh clone needs
`git submodule update --init External/googletest` before configuring (disable tests with
`-DTT_BUILD_TESTS=OFF` to skip it). The `External/glog` google/glog submodule in
`.gitmodules` is not referenced by any CMakeLists and can be ignored (it only matters if you
switch CMake back to real glog).

### Configure & build

```shell
cd TinyTorch
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Equivalent single-command form:

```shell
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Run the sample

```shell
./build/bin/tt_sample        # or: cd build && ./bin/tt_sample
```

Expected output: 50 Adam optimization steps with decreasing loss
(`Step 0 Loss: ~7.4` down to `Step 50 Loss: ~0.008`).

### Run the tests

```shell
cd build && ctest --output-on-failure   # or: ./build/bin/test_tiny_torch_all to run the binary directly
```

Every gtest case is registered as its own ctest entry (suite names carry a `TinyTorch`
prefix, e.g. `TinyTorchSliceOps`); use `./build/bin/test_tiny_torch_all --gtest_filter='...'`
to run a subset (e.g. `--gtest_filter='AllDevices/TinyTorchSliceOps*'` or
`--gtest_filter='*-CPU'`).

### Build options (CMake)

| Option               | Default | Effect                                                        |
|----------------------|---------|---------------------------------------------------------------|
| `TT_WITH_CUDA`       | ON      | Try `find_package(CUDAToolkit 11.8)`; auto-disables if absent |
| `TT_BUILD_SAMPLES`   | ON      | Build `tt_sample`                                              |
| `TT_BUILD_TESTS`     | ON      | Build `test_tiny_torch_all` (needs the googletest submodule) + ctest |
| `TT_ALL_OUTPUT_TO_BIN` | OFF   | Put the library into `build/bin` as well                     |

CPU-only build: `-DTT_WITH_CUDA=OFF` (or simply don't have CUDA installed).
CUDA build: set the toolkit explicitly (it is not on PATH here):
`cmake -S . -B build-cuda -DCUDAToolkit_ROOT=/home/dari/voxray/reconstruction/dependencies_linux/cuda_12_8 -DCMAKE_CUDA_COMPILER=<...>/cuda_12_8/bin/nvcc`
then `cmake --build build-cuda -j$(nproc)`.
CUDA architectures are resolved dynamically in the top-level `CMakeLists.txt` via
`cmake/select_compute_arch.cmake` (copied from saiga). If a parent project (e.g.
voxray-reconstruction) sets `TT_CUDA_ARCH`, TinyTorch uses it so the `torch` target and the
rest of the build target the same archs; otherwise the installed GPUs are autodetected.
Each resolved arch is compiled both as real code (`sm_N`) and virtual/PTX (`N-virtual`),
so the library also runs on slightly newer GPUs. (The local GPU is an RTX 5080, sm_120.)

### Build outputs

- `build/src/libtorch.so` — the library (target name `torch`)
- `build/src/include/torch/tiny_torch_build_config.h` — generated config (`TT_HAS_CUDA`)
- `build/bin/tt_sample` — sample executable
- `build/bin/test_tiny_torch_all` — unit-test executable (each gtest case registered as its own `ctest` entry)

### Linking your own code

```cmake
find_package(TinyTorch)        # or add_subdirectory(TinyTorch)
target_link_libraries(your_target torch)
```

Include with `#include "torch/tiny_torch.h"`; the public include dirs are `src/`
and the build-generated `build/src/include`.

## Code Conventions

- Formatting: `.clang-format` (Google-based): 4-space indent, **Allman braces**, 120-column
  limit, left-aligned pointers (`int* x`), sorted/regrouped includes.
  Run: `clang-format -i <file>` after edits.
- Every source file starts with the MIT copyright header block
  (`Copyright (c) 2022 Darius Rückert / Licensed under the MIT License.`).
- Namespace `tinytorch` everywhere; never use `using namespace tinytorch;` in headers.
- `TINYTORCH_API` on every symbol exposed across the shared library boundary (hidden
  visibility on Linux).
- Error handling: use `CHECK` / `CHECK_EQ` / `CHECK_LT` / ... from `glog/logging.h`
  (vendored tiny-glog). On failure it prints `Check failed in <file>:<line>` and calls
  `g_custom_glog_fail_func` (default: `abort()`; overridable via `google::InstallFailureFunction`).
  There is no custom exception-throwing style; `TinyTorchException` exists in
  tiny_torch_config.h but is rarely used.
- Debug vs release: `TT_DEBUG` = 1 unless `NDEBUG`. Debug enables the (slow)
  `CUDA_KERNEL_ASSERT` bounds checks in device code; release compiles them out.
- CPU kernels: parallelize with `#pragma omp parallel for num_threads(get_num_threads())`;
  thread count is set via `tinytorch::set_num_threads(n)` (thread-local).
- CUDA kernels: guard device work with `cuda::DeviceGuard guard(device);`, check runtime
  calls with `TT_CHECK_CUDA_ERROR`, and bounds-check with `CUDA_KERNEL_ASSERT` (debug only).
- Scalar-type switches: always end with the `default: CHECK(false) << "invalid input type"` case.
- Mixed host/device code goes through `TT_HD` / `TT_DEVICE_CODE` macros — do not write raw
  `__host__ __device__`.
- 32-bit vs 64-bit indexing: host kernels use 64-bit indices (`TensorInfo`), device kernels
  default to 32-bit (`TensorInfoCuda`) with a runtime `is_32bit_addressable()` check.

## Gotchas

- The graph is **rebuilt every forward and destroyed after backward** — holding onto edges or
  calling `backward()` twice without a new forward will fail.
- `backward()` requires the loss to `requires_grad()` and to have an edge
  (`CHECK(loss.getEdge())`); a numel-1 loss gets an implicit grad of 1.
- `SGDOptimizer` uses capitalized `Step()`/`ZeroGrad()` while `Adam` uses `step()`/`zero_grad()`.
- `ops_functions.cpp:261` emits a `-Wreturn-type` warning in CPU-only builds
  (`ToDeviceNode::forward` has no return in the `#else` branch after `CHECK(false)`);
  it is harmless but expected — don't "fix" it by changing behavior.
- `External/glog` (google/glog submodule) is dead weight for the build; `External/tiny-glog`
  is the glog replacement actually compiled and linked (`glog::glog` is an alias to `tiny-glog`).
- `src/CMakeLists.txt` globs sources (`FILE(GLOB_RECURSE *.cpp *.cu)`) — **new .cpp/.cu files
  are picked up only after re-running CMake** (re-configure the build dir).
- `torch` is a SHARED library; on Windows builds DLL-export warnings are suppressed, on Linux
  visibility is hidden — forgetting `TINYTORCH_API` = link errors in consumers.
- `nn::functional::grid_sample`, `conv2d`, `matmul` exist; weight saving/loading (`save`/`load`)
  throws `std::runtime_error("not implemented")`.
- `IValue`-based non-tensor args: nodes receive them typed (e.g. `IValue beta`), so
  `softplus(Tensor, double)`'s backward returns **two** entries (tensor grad + dummy for beta)
  — keep the counts aligned with `num_inputs_of_forward`.

## Verification

```shell
cmake -S . -B build && cmake --build build -j$(nproc)   # must compile cleanly
cd build && ctest --output-on-failure                   # all unit tests must pass
./build/bin/tt_sample                                     # loss must decrease monotonically
```

When CUDA is available, also build and test the CUDA variant (see the CUDA build command above):
`cmake -S . -B build-cuda ... && cmake --build build-cuda -j$(nproc) && cd build-cuda && ctest`.
The suite is the same binary; the `/CUDA` parameterizations run there and are auto-skipped on the
CPU-only build.

Acceptance criteria: a clean build, a fully-passing `ctest` run, and a decreasing-loss
sample run. There is no linter config or CI.

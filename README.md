# vnm

## Overview
`vnm` is a minimalist, compiled neural network library written in the V programming language. It is designed for multi-variable regression and classification tasks, focusing on resource-constrained micro-architectures, mobile environments (such as Termux), embedded systems (IoT), and desktop platforms.

By bypassing automatic garbage collection and utilizing explicit manual memory management, `vnm` compiles directly to native C. This minimizes runtime engine overhead, producing incredibly small binaries that enable low-latency single-vector inference and training directly on CPU cores.

---

## Key Features

*   **Hybrid C-Interop Architecture:** Integrates platform-specific C helper routines (`vnm_arm64.c`) compiled alongside V source files. On ARM64 platforms (Apple Silicon, Raspberry Pi, Android/Termux), this bypasses V's lexical compiler scanner constraints to directly compile ARM NEON SIMD intrinsics (such as `vmlaq_f32` and `vaddvq_f32` in `neon_dot_product_arm64`), accelerating matrix-vector calculations.
*   **Low-Latency Inference:** The inference path (where `cols_b == 1`) is optimized to use hardware-specific SIMD dot-products directly. On modern ARM64 CPUs, a single forward pass for a standard compact model typically executes in the sub-50 microsecond range, allowing for thousands of inferences per second.
*   **Micro-Binary & L1 I-Cache Efficiency:** Unlike heavy frameworks (e.g., TFLite, ONNX), `vnm` has **zero external dependencies** and no computational graph parsing overhead. The resulting stripped binary is typically under 300 KB. This allows the entire inference engine to fit inside the CPU's L1 Instruction Cache (I-Cache), reducing RAM bottlenecks and minimizing cold-start latencies.
*   **Zero-Overhead Silent Mode:** Supports a compile-time `-d vnm_silent` flag that strips out standard output, I/O operations, and dynamic string allocations/interpolations during hot loops, dedicating 100% of CPU cycles to pure math.
*   **Compiler-Friendly Loop Structure:** Dot-product operations in the sequential matrix multiplication routines avoid manual, hardcoded loop unrolling. Instead, they use simple sequential loops, allowing C compiler backends (GCC/Clang) to leverage native SIMD vectorization (SSE, AVX2, AVX-512) and issue Fused Multiply-Add (FMA) instructions where applicable.
*   **Fast Inverse Square Root (Software & Hardware):** Features multiple fast reciprocal square root implementations for the Adam optimizer. On ARM64 architectures, it can utilize native hardware-assisted instructions (`frsqrte` with GCC-compatible `%w` 32-bit register operand constraints in C inline assembly). For other architectures, it falls back to a software-level floating-point bit-manipulation hack (Quake III approach) implemented in C.
*   **Schraudolph's Exponential Approximation:** Integrates Schraudolph's floating-point bit-manipulation algorithm for fast $e^x$ approximation in C (`fast_exp_c`). This speeds up the evaluation of complex transcendental mathematical activations like `approx_sigmoid` and `approx_tanh`.
*   **Branchless Forward ReLU:** Employs standard hardware-level maximum operations (`fmaxf` via `fast_max_neon`) that map to native, branchless CPU instructions (such as `maxss` or `fmax`), reducing potential pipeline stalls caused by branch mispredictions.
*   **Minimized Runtime Allocations:** Once the neural network and its layers are initialized, the training loop avoids dynamic memory allocations (such as `malloc` or V array resizing) inside hot paths, which helps maintain deterministic execution latency.
*   **Workload Distribution:** Distributes row-wise matrix multiplications across multiple CPU threads using V's native `spawn` keyword, querying physical core counts via `runtime.nr_cpus()`.
*   **Fallback Thresholds:** To prevent scheduling and context-switching overhead on smaller workloads, the library routes execution to a single-threaded path if the matrix has fewer than 64 rows, or during single-vector calculations (where `cols_b == 1`).
*   **Advanced Optimizers (SGD & Adam):** Native support for both standard Stochastic Gradient Descent (SGD) and the **Adam** optimizer, featuring momentum, velocity, and bias correction (`beta1`, `beta2`) for stable convergence.
*   **V Bounds Checking Bypass Option:** Supports bypassing V's implicit array bounds checking inside hot training loops and layer operations. By utilizing unsafe pointer indexing (`&training_inputs[0]`), the compiler translates loops directly into standard C pointer arithmetic.
*   **Loop Fusion:** Evaluates Mean Squared Error (MSE) loss and calculates the output layer's gradient delta ($\delta$) simultaneously in a single fused loop, reducing cache sweeps and memory bus traffic.
*   **Flexible Compile-Time Precision (`Real` Alias):** Features compile-time precision mapping. Single-precision `f32` is employed as the default mode to maximize SSE2/NEON vectorization throughput and reduce memory bandwidth requirements. Passing `-d vnm_f64` at compile-time promotes the engine to double-precision `f64`.
*   **Consistent API Boundaries:** To preserve clean syntax, creator APIs (`vector`, `scalar`, `new_tensor`) accept standard V float arrays (`[]f64`). The library automatically maps these inputs to the active engine precision (`Real`) during tensor instantiation.
*   **RNN & Sequence Modeling Support:** Features native support for Recurrent Neural Networks (RNN). Layers can maintain hidden states and recurrent weights across sequence steps, enabling time-series and sequential data processing.
*   **Dropout Regularization:** Includes a dropout mechanism with drop masks to randomly deactivate neurons during the training phase, helping prevent overfitting in complex architectures.
*   **Extensible Custom Activation Interface:** Provides a modular API (`CustomActivation`) allowing developers to pass custom forward mathematical functions and their corresponding derivatives via function pointers at runtime (e.g., LeakyReLU, ELU), ensuring high extensibility without modifying the core library.
*   **Symmetric Weight Quantization (INT8 & INT16):** Integrates post-training symmetric quantization. It dynamically analyzes weight scales per layer to compress floating-point parameters to discrete `int8` or `int16` ranges, enabling memory footprint reduction and fast integer arithmetic mapping on edge hardware.
*   **Custom Binary Model Serialization (Save/Load):** Fully trained models—including topology, weights, biases, optimizer settings, quantization mode, and normalization parameters (`means`, `stds`)—are serialized directly into a custom-structured binary format (`VNMB`). This avoids the parsing and memory overhead of text-based formats like JSON, maintaining an extremely small disk footprint.
*   **Z-Score Input Normalization:** Features built-in Z-Score input standardization and Target Min-Max scaling. The model computes, stores, and serializes feature means, standard deviations, and boundaries during training, applying them to future predictions.
*   **Compile-Time Conditional Safety (`-d vnm_safe`):** Dual-mode compilation ensures versatility. You can compile with size-validation, dimension mismatch checks, and zero-division assertions during development, or completely prune these assertions at compile-time for minimized runtime overhead in production.
*   **He (Kaiming) Initialization:** Built-in uniform random initialization ($\sqrt{6 / n_{\text{in}}}$) optimized specifically for stable training, preventing gradient explosion and dead neurons.
*   **IKJ Loop Order:** Matrix multiplication implements an optimized IKJ loop order. This naturally accesses both matrix operands contiguously (stride-1), reducing the need for matrix transposition.

---

## Technical Specifications

### Core Architecture
1.  **Tensor & Matrix Layer:** Features a multi-dimensional `Tensor` interface and flat `Matrix` layouts utilizing raw pointers, flexible precision `Real` arrays, and manual memory deallocation routines.
2.  **Sequential API:** Employs a linear configuration interface (`model.add()`) to specify layers, mapping input/output sizes, activation types, dropout rates, and RNN toggles.
3.  **Configurable Activations:** Supports multiple distinct activation functions (`sigmoid`, `relu`, `tanh`, `linear`, and `custom`), automatically handling their derivatives during backpropagation.

---

## Installation

### 1. As a V Module (Recommended)
You can install `vnm` directly as a dependency into your global V modules using the V package manager:
```bash
v install --git https://github.com/tailsmails/vnm
```
Once installed, you can import it into any V project:
```v
import vnm
```

### 2. Manual Installation (Local Copy)
If you prefer to include the source files directly within your project directory:
1. Clone the repository into your project path.
2. Put the `vnm_arm64.c` helper file alongside `vnm.v` in your local directory.
3. Import the module locally (`import vnm`).
4. Compile your project using optimized compilation flags:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main
```
*Note: Do not compile it with tcc (`v main.v`) if your device is ARM64!*

---

## Minimal Code Example

Here is a comprehensive example demonstrating how to define a custom activation function, train a network, serialize the trained state to a binary file, deserialize it back, apply symmetric INT8 quantization, and run low-latency predictions:

```v
import vnm
import math
import os

// 1. Define custom activation functions (e.g., LeakyReLU)
fn my_leaky_relu(x vnm.Real) vnm.Real {
	return if x > 0 { x } else { vnm.Real(0.01) * x }
}

fn my_leaky_relu_derivative(y vnm.Real) vnm.Real {
	return if y > 0 { vnm.Real(1.0) } else { vnm.Real(0.01) }
}

fn main() {
	mut inputs := []vnm.Tensor{}
	mut targets := []vnm.Tensor{}

	// Training Dataset (logical OR gate)
	inputs << vnm.vector([0.0, 0.0])
	targets << vnm.vector([0.0])

	inputs << vnm.vector([0.0, 1.0])
	targets << vnm.vector([1.0])

	inputs << vnm.vector([1.0, 0.0])
	targets << vnm.vector([1.0])

	inputs << vnm.vector([1.0, 1.0])
	targets << vnm.vector([1.0])
	
	println('Building model architecture with Custom Activation...')
	mut model := vnm.new_sequential(.adam)
	
	leaky_act := vnm.CustomActivation{
		forward: my_leaky_relu
		derivative: my_leaky_relu_derivative
	}
	
	// Add custom activation layer
	model.add_custom(2, 4, leaky_act, 0.0, false)
	model.add(4, 1, .sigmoid, 0.0, false)
	model.set_normalize(true)
	
	println('Starting training (600 epochs)...')
	model.train(inputs, targets, 600, 0.05) or {
		println('Training failed: ${err}')
		return
	}
	
	// Serialize model state to disk in compact binary format
	model_file := 'or_model.vnm'
	model.save(model_file) or { panic(err) }
	println('Model saved to ${model_file} (Binary Format)')

	// Load model to verify Binary Deserialization
	mut loaded_model := vnm.load_sequential(model_file) or { panic(err) }
	
	// Restore custom activation function pointers (cannot be serialized)
	loaded_model.net.layers[0].custom_act = leaky_act

	// Apply Post-Training Symmetric INT8 Quantization
	loaded_model.quantization = .int8
	loaded_model.apply_quantization()
	println('Symmetric INT8 Quantization successfully applied.')

	test_cases := [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	]

	println('\nRunning Predictions (using loaded & quantized model):')
	for test in test_cases {
		pred_tensor := loaded_model.predict(vnm.vector(test)) or { continue }
		output := pred_tensor.get(0)
		expected := if test[0] > 0.0 || test[1] > 0.0 { 1.0 } else { 0.0 }
		
		println(' Input: [${test[0]:.1f}, ${test[1]:.1f}]  =>  Predicted Probability: ${output:6.4f}  |  Expected: ${expected:.1f}')
		pred_tensor.free()
	}
	
	// Manual Memory Cleanup
	model.free()
	loaded_model.free()
	for mut t in inputs { t.free() }
	for mut t in targets { t.free() }
	os.rm(model_file) or {}
}
```

---

## Quick Start (Build & Run)

You can clone, compile, and run the project inside standard environments (including Android Termux) using:

```bash
pkg update -y && pkg install -y git clang make && if ! command -v v >/dev/null 2>&1; then git clone --depth=1 https://github.com/vlang/v && cd v && make && ./v symlink && cd ..; fi && git clone --depth=1 https://github.com/tailsmails/vnm && cd vnm && v -cc clang -prod -d no_bounds_checking main.v -o main && ./main
```

---

## Compilation Modes

### 1. Safe & Debug Mode
Compiles with compile-time conditional checks, bounds/dimension assertions, and verbose validation logging (defaults to default `f32` precision):
```bash
v -d vnm_safe -cc clang main.v -o main && ./main
```

### 2. Single-Precision (f32) Production Mode (Default)
Strips away all assertions, logs, and dimension checks at compile-time, maximizing compiler-level loop unrolling and SSE2/NEON hardware vectorization:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main && ./main
```

### 3. Double-Precision (f64) Production Mode
Promotes precision to double-precision `f64` for backward compatibility or strict high-precision simulations, while keeping all runtime allocation constraints intact:
```bash
v -d vnm_f64 -cc clang -prod -d no_bounds_checking main.v -o main && ./main
```

### 4. Maximum Performance / Silent Mode
Completely strips out all standard I/O (like `println`) and string allocations during runtime. Ideal for raw edge-inference, game engines, or Real-Time systems where determinism and zero-overhead are required:
```bash
v -cc clang -prod -d no_bounds_checking -d vnm_silent main.v -o main && ./main
```

---

## Requirements
*   **Operating System:** Cross-platform (Linux, Android/Termux, Windows, macOS).
*   **Compiler:** V programming language compiler.
*   **C Compiler backend:** Clang or GCC (required for host-vectorization flags).

---

## Log Interpretation
*   **Mean Squared Loss:** Shows the mean squared error (MSE) value computed over the training dataset.
*   **Active LR:** Displays the current learning rate value, reflecting any step or exponential decay applied during training.
*   **BLIND Generalization Test:** Evaluates the trained model on unseen parameters outside the training grid, printing the absolute error compared to the analytical expectations.

---

## Disclaimer
This library is developed for educational purposes, academic research, and edge-computing applications requiring lightweight, dependency-free machine learning implementations.

---

## License
![License](https://img.shields.io/badge/License-MIT-red.svg)

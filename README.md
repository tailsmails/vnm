# vnm

## Overview
`vnm` is an extremely lightweight, compiled neural network library written in pure V. It is designed for multi-variable regression and classification tasks, optimized for low-latency execution on resource-constrained micro-architectures, mobile environments (such as Termux), and high-performance desktop platforms.

By bypassing automatic garbage collection and utilizing explicit manual memory management, `vnm` compiles directly to native C. This approach minimizes runtime engine overhead, enabling high-throughput training and inference directly on standard CPU cores.

---

## Key Features

*   **Advanced Optimizers (SGD & Adam):** Built-in support for both standard Stochastic Gradient Descent (SGD) and the highly efficient **Adam** optimizer. It features momentum, velocity, and bias correction (`beta1`, `beta2`) for rapid and stable convergence.
*   **RNN & Sequence Modeling Support:** Beyond standard feed-forward networks, `vnm` features native support for Recurrent Neural Networks (RNN). Layers can maintain hidden states and recurrent weights across sequence steps, enabling time-series and sequential data processing.
*   **Dropout Regularization:** Includes a highly optimized dropout mechanism with drop masks. It randomly deactivates neurons during the training phase and automatically scales active neurons, effectively preventing overfitting in complex architectures.
*   **JSON Model Serialization (Save/Load):** Fully trained models—including network topology, weights, biases, and normalization parameters—can be serialized and deserialized to/from disk via standard JSON configuration with zero external dependencies.
*   **Zero-Configuration Auto-Normalization:** Features built-in Z-Score input standardization and Target Min-Max scaling. The model automatically computes, stores, and serializes feature means, standard deviations, and boundaries during training, applying them seamlessly to future predictions with zero user configuration.
*   **Compile-Time Conditional Safety (`-d vnm_safe`):** Dual-mode compilation ensures maximum versatility. You can compile with size-validation, dimension mismatch checks, and zero-division assertions during development, or completely prune these assertions at compile-time for absolute zero-overhead execution in production.
*   **Flexible Compile-Time Precision (`Fnn` Alias):** Seamlessly toggles between Double-Precision `f64` (default) and Single-Precision `f32` (by passing `-d vnm_f32` at compile-time). This allows you to double your SIMD (AVX/NEON) vectorization throughput and cut memory bandwidth requirements in half when needed.
*   **He (Kaiming) Initialization:** Built-in uniform random initialization ($\sqrt{6 / n_{\text{in}}}$) optimized specifically for stable training, preventing gradient explosion and dead neurons.
*   **Manual Memory Control:** Bypasses garbage collection latency. It utilizes explicit manual memory freeing (`.free()`) on tensors, matrices, and layers to maintain a low, highly predictable memory footprint.
*   **Zero-Transpose GEMM via IKJ Layout:** Parallel matrix multiplication implements an optimized IKJ loop order. This naturally accesses both matrix operands contiguously (stride-1), entirely eliminating the need for matrix transposition and saving RAM allocation overhead.
*   **Advanced Cache Tiling (Loop Blocking):** Large matrix multiplications are partitioned into optimized cache-resident tiles ($64 \times 64$). This prevents CPU cache thrashing and keeps core data close to the arithmetic logic units.
*   **Hybrid MatMul Dispatch:** Employs a dual-mode dispatch mechanism. It uses register-allocated, low-latency IJK operations for small calculations and scales to tiled IKJ multi-threaded parallel execution (`matmul_parallel`) using host threads (`runtime.nr_jobs()`) for larger matrix workloads.
*   **Compiler-Level Vectorization:** Integrates compiler-level flags (`#flag -O3`, `-ffast-math`, `-march=native`, `-funroll-loops`) directly within V source files, instructing the compiler backend (GCC/Clang) to generate vectorized SIMD instructions (AVX/NEON).

---

## Technical Specifications

### Core Architecture
1.  **Tensor & Matrix Layer:** Features a multi-dimensional `Tensor` interface and flat `Matrix` layouts utilizing raw pointers, flexible precision `Fnn` arrays, and manual memory deallocation routines.
2.  **Sequential API:** Employs a linear configuration interface (`model.add()`) to specify layers, mapping input/output sizes, activation types, dropout rates, and RNN toggles effortlessly.
3.  **Configurable Activations:** Supports multiple distinct activation functions (`sigmoid`, `relu`, `tanh`, and `linear`), automatically handling their derivatives during backpropagation.

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
2. Import the module locally (`import vnm`).
3. Compile your project using optimized compilation flags:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main
```

---

## Minimal Code Example

Here is how easily you can define, train, save, and run predictions using `vnm` and its Adam optimizer:

```v
import vnm
import time
import math
import os

fn main() {
	mut inputs := []vnm.Tensor{}
	mut targets := []vnm.Tensor{}

	// XOR Dataset
	inputs << vnm.vector([0.0, 0.0])
	targets << vnm.vector([0.0])

	inputs << vnm.vector([0.0, 1.0])
	targets << vnm.vector([1.0])

	inputs << vnm.vector([1.0, 0.0])
	targets << vnm.vector([1.0])

	inputs << vnm.vector([1.0, 1.0])
	targets << vnm.vector([0.0])
	
	println('Building model architecture...')
	// Use Adam optimizer for faster convergence
	mut model := vnm.new_sequential(.adam)
	
	// add(input_size, output_size, activation, dropout_rate, is_rnn)
	model.add(2, 8, .relu, 0.0, false)
	model.add(8, 1, .sigmoid, 0.0, false)
	model.set_normalize(false)
	
	println('Starting training (1000 epochs)...')
	sw := time.new_stopwatch()
	
	// train_with_decay(inputs, targets, epochs, learning_rate, decay_rate, decay_steps)
	model.train_with_decay(inputs, targets, 1000, 0.05, 1.0, 0) or {
		println('Training failed: ${err}')
		return
	}

	elapsed := sw.elapsed()
	println('\nTraining completed in: ${elapsed}')
	
	// Save the model state to disk
	model_file := 'xor_model.json'
	model.save(model_file) or { panic(err) }
	println('Model saved to ${model_file}')

	// Load model to verify JSON Serialization
	mut loaded_model := vnm.load_sequential(model_file) or { panic(err) }

	test_cases := [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	]

	println('\nRunning Predictions (using loaded model):')
	for test in test_cases {
		pred_tensor := loaded_model.predict(vnm.vector(test)) or { continue }
		output := pred_tensor.data[0]
		expected := int(test[0]) ^ int(test[1])
		
		rounded_output := int(math.round(output))
		status := if rounded_output == expected { 'PASS' } else { 'FAIL' }
		
		println(' Input: [${int(test[0])}, ${int(test[1])}]  =>  Predicted: ${output:6.4f}  |  Expected: ${expected}  [${status}]')
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
Compiles with compile-time conditional checks, bounds/dimension assertions, and verbose validation logging (defaults to double-precision `f64`):
```bash
v -d vnm_safe -cc clang main.v -o main && ./main
```

### 2. High-Performance Double-Precision (f64) Production Mode
Strips away all assertions, logs, and dimension checks at compile-time, maximizing compiler-level loop unrolling and CPU optimizations using double-precision float arrays:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main && ./main
```

### 3. Ultra-Performance Single-Precision (f32) Production Mode
Compiles with single-precision float arrays to leverage 2x higher vectorization throughput (SIMD) and half the memory bandwidth footprint, delivering maximum execution speed:
```bash
v -d vnm_f32 -cc clang -prod -d no_bounds_checking main.v -o main && ./main
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
*   **BLIND Generalization Test:** Evaluates the trained model on unseen parameters outside the training grid, printing the absolute error compared to the analytical physical/logical expectations.

---

## Disclaimer
This library is developed for educational purposes, academic research, and edge-computing applications requiring lightweight, dependency-free machine learning implementations.

---

## License
![License](https://img.shields.io/badge/License-MIT-blue.svg)
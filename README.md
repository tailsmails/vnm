# vnm

## Overview
`vnm` is a highly lightweight, compiled neural network library written in pure V. It is designed for multi-variable regression and classification tasks, specifically optimized for high-throughput, low-latency execution strictly on CPU architectures. It is ideal for resource-constrained micro-architectures, mobile environments (such as Termux), and high-performance desktop platforms.

By bypassing automatic garbage collection, avoiding large frameworks, and utilizing explicit manual memory management, `vnm` compiles directly to native C. It is built to run exclusively on CPU cores (x86 and ARM), making machine learning accessible and highly efficient without requiring heavy GPU drivers or CUDA/ROCm dependencies.

---

## Key Features

*   **Dynamic Multi-Threaded Execution:** Features dynamic thread dispatching using V's native `spawn` keyword. It queries physical CPU cores via `runtime.nr_cpus()` and divides row-wise matrix multiplications among multiple threads without locking or data races, scaling CPU performance on multi-core processors.
*   **Smart Fallback Thresholds:** Prevents scheduling and context-switching overhead on smaller workloads. The library dynamically routes execution to an optimized single-threaded path if the matrix has fewer than 64 rows, or during single-vector calculations (where `cols_b == 1`).
*   **Advanced Optimizers (SGD & Adam):** Built-in support for both standard Stochastic Gradient Descent (SGD) and the highly efficient **Adam** optimizer. It features momentum, velocity, and bias correction (`beta1`, `beta2`) for rapid and stable convergence.
*   **True Zero-Allocation Training:** Bypasses garbage collection and manual allocator (`malloc`/`free`) overhead. Once initialized, the entire forward pass, backpropagation, and Adam weight update pipeline runs with **exactly zero** dynamic memory allocations, keeping all active data inside pre-allocated cache buffers on the CPU heap.
*   **Multi-Graded Math Approximations:** Provides a flexible, four-tiered architecture for mathematical functions to balance execution speed against gradient precision:
    *   **Precise (Grade 0):** Employs exact IEEE 754 standard CPU math (`math.exp`, `math.tanh`, and standard square root) for high accuracy.
    *   **Balanced (Grade 1):** Utilizes highly accurate Padé rational approximations for activation functions and a 2-iteration fast inverse square root for the optimizer.
    *   **Fast (Grade 2):** Uses the branchless Elliott algebraic approximation ($y = \frac{x}{1+|x|}$) which eliminates slow exponentials, prevents CPU pipeline stalls, and accelerates training via 1-iteration fast inverse square root.
    *   **Extreme (Grade 3):** Replaces activations with piecewise linear/hard representations (Hard-Sigmoid, Hard-Tanh) and uses raw 0-iteration bit-hack inverse square root for maximum raw CPU throughput.
*   **Quake III Fast Inverse Square Root:** Replaces heavy hardware division and square root operations inside the Adam updates. Employing a modified version of the fast inverse square root bit-hack (optimized for both `f32` and `f64` precision), Adam updates are converted into rapid, low-cycle multiplications.
*   **Branchless Activation Functions (ReLU):** To prevent CPU pipeline stalls caused by branch mispredictions, ReLU and its derivatives are written in a completely branchless manner using floating-point bitwise absolute values and sign copies.
*   **V Bounds Checking Bypass:** Bypasses V's implicit array bounds checking inside hot training loops and layer operations. By utilizing unsafe pointer indexing (`&training_inputs[0]`, `&nn.layers[0]`), the compiler translates loops directly into raw, high-performance C pointer arithmetic.
*   **Loop Fusion:** Evaluates Mean Squared Error (MSE) loss and calculates the output layer's gradient delta ($\delta$) simultaneously in a single fused loop. This reduces CPU cache sweeps and significantly lowers memory bus traffic.
*   **Flexible Compile-Time Precision (`Real` Alias):** Features dynamic precision mapping. Single-precision `f32` is employed as the default compile-time mode to maximize SSE2/AVX/NEON vectorization throughput and cut memory bandwidth requirements in half. If high precision is required, passing `-d vnm_f64` at compile-time seamlessly promotes the engine to double-precision `f64`.
*   **Seamless User-Friendly Boundaries:** To preserve clean syntax, creator APIs (`vector`, `scalar`, `new_tensor`) accept standard V float arrays (`[]f64`). The library automatically and internally maps these inputs to the active engine precision (`Real`) during tensor instantiation with negligible overhead.
*   **RNN & Sequence Modeling Support:** Beyond standard feed-forward networks, `vnm` features native support for Recurrent Neural Networks (RNN). Layers can maintain hidden states and recurrent weights across sequence steps, enabling time-series and sequential data processing on CPU cores.
*   **Dropout Regularization:** Includes a highly optimized dropout mechanism with drop masks. It randomly deactivates neurons during the training phase and automatically scales active neurons, effectively preventing overfitting in complex architectures.
*   **JSON Model Serialization (Save/Load):** Fully trained models—including network topology, weights, biases, and normalization parameters—can be serialized and deserialized to/from disk via standard JSON configuration with zero external dependencies.
*   **Zero-Configuration Auto-Normalization:** Features built-in Z-Score input standardization and Target Min-Max scaling. The model automatically computes, stores, and serializes feature means, standard deviations, and boundaries during training, applying them seamlessly to future predictions with zero user configuration.
*   **Compile-Time Conditional Safety (`-d vnm_safe`):** Dual-mode compilation ensures maximum versatility. You can compile with size-validation, dimension mismatch checks, and zero-division assertions during development, or completely prune these assertions at compile-time for absolute zero-overhead execution in production.
*   **He (Kaiming) Initialization:** Built-in uniform random initialization ($\sqrt{6 / n_{\text{in}}}$) optimized specifically for stable training, preventing gradient explosion and dead neurons.
*   **Zero-Transpose GEMM via IKJ Layout:** Matrix multiplication implements an optimized IKJ loop order. This naturally accesses both matrix operands contiguously (stride-1), maximizing CPU L1/L2 cache locality, increasing cache hits, and entirely eliminating the need for matrix transposition.

---

## Technical Specifications

### Core Architecture
1.  **Tensor & Matrix Layer:** Features a multi-dimensional `Tensor` interface and flat `Matrix` layouts utilizing raw pointers, flexible precision `Real` arrays, and manual memory deallocation routines.
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

	// XOR Dataset (accepts standard []f64 literals natively)
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
	model.set_approx_level(.fast)
	
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
Compiles with compile-time conditional checks, bounds/dimension assertions, and verbose validation logging (defaults to default `f32` precision):
```bash
v -d vnm_safe -cc clang main.v -o main && ./main
```

### 2. Ultra-Performance Single-Precision (f32) Production Mode (Default)
Strips away all assertions, logs, and dimension checks at compile-time, maximizing compiler-level loop unrolling and SSE2/AVX/NEON CPU hardware vectorization:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main && ./main
```

### 3. High-Performance Double-Precision (f64) Production Mode
Promotes precision to double-precision `f64` for backward compatibility or strict high-precision simulations, while keeping all zero-allocation engine speedups intact:
```bash
v -d vnm_f64 -cc clang -prod -d no_bounds_checking main.v -o main && ./main
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
This library is developed for educational purposes, academic research, and edge-computing applications requiring lightweight, dependency-free CPU-only machine learning implementations.

---

## License
![License](https://img.shields.io/badge/License-MIT-blue.svg)

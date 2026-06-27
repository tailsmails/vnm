# vnm

## Overview
`vnm` is an extremely lightweight, compiled neural network library written in pure V. It is designed for multi-variable regression and classification tasks, optimized for low-latency execution on resource-constrained micro-architectures, mobile environments (such as Termux), and high-performance desktop platforms.

By bypassing automatic garbage collection and utilizing explicit manual memory management, `vnm` compiles directly to native C. This approach minimizes runtime engine overhead, enabling high-throughput training and inference directly on standard CPU cores.

---

## Key Features

*   **Zero-Configuration Auto-Normalization:** Features built-in Z-Score input standardization. The model automatically computes, stores, and serializes feature means and standard deviations during training, applying them seamlessly to future predictions with zero user configuration.
*   **Compile-Time Conditional Safety (`-d vnm_safe`):** Dual-mode compilation ensures maximum versatility. You can compile with size-validation and verbose logging during development, or completely prune these assertions at compile-time for absolute zero-overhead execution in production.
*   **He (Kaiming) Initialization:** Built-in uniform Kaiming initialization ($\sqrt{6 / n_{\text{in}}}$) optimized specifically for stable training of deep ReLU hidden layers, preventing gradient explosion and dead neurons.
*   **Manual Memory Control:** Bypasses garbage collection latency. It utilizes explicit manual memory freeing (`.free()`) on tensors, matrices, and layers to maintain a low, highly predictable memory footprint.
*   **Inplace Transpose-Free Operations:** Computes operations requiring transposed matrices ($A^T \cdot B$ and $A \cdot B^T$) directly via pointer arithmetic and stride adjustments, completely avoiding RAM-copy transpose overhead.
*   **Hybrid MatMul Dispatch:** Dynamically switches between single-threaded inline pointer multiplication for smaller matrices and multi-threaded parallel multiplication (`matmul_parallel`) using host threads (`runtime.nr_jobs()`) for larger matrices.
*   **Compiler-Level Vectorization:** Integrates compiler-level flags (`#flag -O3`, `-ffast-math`, `-march=native`, `-funroll-loops`) directly within V source files, instructing the compiler backend (GCC/Clang) to generate vectorized SIMD instructions (AVX/NEON).

---

## Technical Specifications

### Core Architecture
1.  **Tensor & Matrix Layer:** Features a multi-dimensional `Tensor` interface and flat `Matrix` layouts utilizing raw pointers and manual memory deallocation routines.
2.  **Sequential API:** Employs a linear configuration interface (`model.add()`) to specify layers, managing under-the-hood weight initialization and activation binding.
3.  **Configurable Activations:** Supports multiple distinct activation functions (`sigmoid`, `relu`, `tanh`, and `linear`) configurable per individual layer.

---

## Installation

### 1. As a V Module (Recommended)
You can install `vnm` directly as a dependency into your global V modules using the V package manager:
```bash
v install --git https://github.com/tailsmails/vnm
```
Once installed, you can import it into any V project:
```v
import tailsmails.vnm
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

Here is how easily you can define, train, save, and run predictions using `vnm` (change the import path to `tailsmails.vnm` if installed as a module):

```v
import vnm // Use "import tailsmails.vnm" if installed via v install

fn main() {
    // 1. Create a Sequential model with the Adam optimizer
    mut model := vnm.new_sequential(.adam)

    // 2. Enable automatic Z-Score standardization (enabled by default)
    model.set_normalize(true)

    // 3. Define the topology (ReLU hidden layers, Linear output layer)
    model.add(3, 128, .relu)
    model.add(128, 64, .relu)
    model.add(64, 3, .linear)

    // 4. Generate or load training data as vnm.Tensor vectors
    mut inputs := []vnm.Tensor{}
    mut targets := []vnm.Tensor{}

    inputs << vnm.vector([1500.0, 4.0, 30.0])
    targets << vnm.vector([0.2, 0.5, 0.1])

    // 5. Train the network with learning rate decay
    // Parameters: inputs, targets, epochs, initial_lr, decay_rate, decay_steps
    model.train_with_decay(inputs, targets, 1500, 0.0008, 0.94, 200)!

    // 6. Predict with automatic input feature scaling
    prediction := model.predict(vnm.vector([2000.0, 3.5, 30.0]))!
    println("Prediction output: \${prediction.data}")

    // 7. Save model configuration and standardization parameters to JSON
    model.save("rocket_model.vnm")!

    // 8. Explicitly deallocate model and tensor memory to prevent leaks
    prediction.free()
    model.free()
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
Compiles with compile-time conditional checks, bounds/dimension assertions, and verbose validation logging:
```bash
v -d vnm_safe -cc clang main.v -o main && ./main
```

### 2. Aggressively Fast Production Mode
Strips away all assertions, logs, and dimension checks at compile-time, maximizing compiler-level loop unrolling and CPU optimizations:
```bash
v -cc clang -prod -d no_bounds_checking main.v -o main && ./main
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
*   **BLIND Generalization Test:** Evaluates the trained model on unseen parameters outside the training grid, printing the absolute error compared to the analytical physics equations.

---

## Disclaimer
This library is developed for educational purposes, academic research, and edge-computing applications requiring lightweight machine learning implementations.

---

## License
![License](https://img.shields.io/badge/License-MIT-blue.svg)
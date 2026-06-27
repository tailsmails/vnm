module vnm

import math
import rand
import runtime
import os
import json

#flag -O3
#flag -ffast-math
#flag -march=native
#flag -funroll-loops

pub enum ActivationType {
	sigmoid
	relu
	tanh
	linear
}

pub enum OptimizerType {
	sgd
	adam
}

pub struct Tensor {
pub:
	shape []int
pub mut:
	data []f64
}

pub fn new_tensor(shape []int, data []f64) Tensor {
	total_size := get_shape_size(shape)
	mut actual_data := data.clone()
	if data.len < total_size {
		actual_data = []f64{len: total_size, init: 0.0}
	}
	return Tensor{
		shape: shape
		data: actual_data
	}
}

pub fn vector(data []f64) Tensor {
	return Tensor{
		shape: [data.len, 1]
		data: data.clone()
	}
}

pub fn scalar(val f64) Tensor {
	return Tensor{
		shape: [1, 1]
		data: [val]
	}
}

@[inline]
fn get_shape_size(shape []int) int {
	mut size := 1
	for dim in shape {
		size *= dim
	}
	return size
}

pub fn (t Tensor) reshape(new_shape []int) !Tensor {
	if _unlikely_(get_shape_size(new_shape) != t.data.len) {
		return error("Cannot reshape Tensor: Total elements mismatch.")
	}
	return Tensor{
		shape: new_shape
		data: t.data.clone()
	}
}

pub fn (t Tensor) flatten() Tensor {
	return Tensor{
		shape: [t.data.len]
		data: t.data.clone()
	}
}

pub fn (t &Tensor) free() {
	unsafe { t.data.free() }
}

pub struct Matrix {
pub:
	rows int
	cols int
pub mut:
	data []f64
}

pub fn new_matrix(rows int, cols int) Matrix {
	return Matrix{
		rows: rows
		cols: cols
		data: []f64{len: rows * cols, init: 0.0}
	}
}

pub fn new_random_matrix(rows int, cols int) Matrix {
	mut m := new_matrix(rows, cols)
	boundary := math.sqrt(6.0 / f64(cols))
	for i in 0 .. m.data.len {
		m.data[i] = rand.f64_in_range(-boundary, boundary) or { 0.0 }
	}
	return m
}

pub fn (m Matrix) transpose() Matrix {
	mut res := new_matrix(m.cols, m.rows)
	unsafe {
		mut ptr_res := &res.data[0]
		for i in 0 .. m.rows {
			offset := i * m.cols
			for j in 0 .. m.cols {
				*ptr_res = m.data[offset + j]
				ptr_res++
			}
		}
	}
	return res
}

pub fn (m &Matrix) free() {
	unsafe { m.data.free() }
}

@[inline; unsafe]
fn matmul_transpose_b_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		mut ptr_res := &res.data[0]
		for i in 0 .. a.rows {
			val_a := a.data[i]
			mut ptr_b := &b.data[0]
			for _ in 0 .. b.rows {
				*ptr_res = val_a * *ptr_b
				ptr_res++
				ptr_b++
			}
		}
	}
}

@[inline; unsafe]
fn matmul_transpose_a_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		mut ptr_res := &res.data[0]
		cols_a := a.cols
		for i in 0 .. cols_a {
			mut sum := 0.0
			mut ptr_a := &a.data[i]
			mut ptr_b := &b.data[0]
			for _ in 0 .. a.rows {
				sum += *ptr_a * *ptr_b
				ptr_a += cols_a
				ptr_b++
			}
			*ptr_res = sum
			ptr_res++
		}
	}
}

@[inline; unsafe]
fn matmul_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		cols_a := a.cols
		cols_b := b.cols
		mut ptr_res := &res.data[0]
		for i in 0 .. a.rows {
			offset_a := i * cols_a
			for j in 0 .. cols_b {
				mut sum := 0.0
				mut ptr_a := &a.data[offset_a]
				mut ptr_b := &b.data[j]
				for _ in 0 .. cols_a {
					sum += *ptr_a * *ptr_b
					ptr_a++
					ptr_b += cols_b
				}
				*ptr_res = sum
				ptr_res++
			}
		}
	}
}

@[manualfree; unsafe]
pub fn matmul_parallel(a Matrix, b Matrix) !Matrix {
	if _unlikely_(a.cols != b.rows) {
		return error("Matrix dimensions mismatch: ${a.cols} vs ${b.rows}")
	}
	mut result := new_matrix(a.rows, b.cols)
	
	if _likely_(a.rows < 64 || b.cols < 64) {
		matmul_inplace(a, b, mut result)
		return result
	}

	b_t := b.transpose()
	num_cores := runtime.nr_jobs()
	threads_count := if a.rows < num_cores { a.rows } else { num_cores }

	mut threads := []thread{}
	rows_per_thread := a.rows / threads_count

	for t in 0 .. threads_count {
		start_row := t * rows_per_thread
		mut end_row := (t + 1) * rows_per_thread
		if t == threads_count - 1 {
			end_row = a.rows
		}
		worker := MatMulWorker{
			a: &a
			b_t: &b_t
			res_data: unsafe { &result.data[0] }
			start_row: start_row
			end_row: end_row
		}
		threads << go worker.run()
	}
	threads.wait()
	
	b_t.free()
	return result
}

struct MatMulWorker {
	a         &Matrix
	b_t       &Matrix
	res_data  &f64
	start_row int
	end_row   int
}

fn (w MatMulWorker) run() {
	unsafe {
		cols_a := w.a.cols
		cols_b := w.b_t.rows
		for i in w.start_row .. w.end_row {
			offset_a := i * cols_a
			for j in 0 .. cols_b {
				offset_b := j * cols_a
				mut sum := 0.0
				for k in 0 .. cols_a {
					sum += w.a.data[offset_a + k] * w.b_t.data[offset_b + k]
				}
				idx := i * cols_b + j
				ptr := w.res_data + idx
				*ptr = sum
			}
		}
	}
}

@[inline]
fn activate(x f64, act ActivationType) f64 {
	match act {
		.sigmoid { return 1.0 / (1.0 + math.exp(-x)) }
		.relu { return if _likely_(x > 0) { x } else { 0.0 } }
		.tanh { return math.tanh(x) }
		.linear { return x }
	}
}

@[inline]
fn activate_derivative(activated_val f64, act ActivationType) f64 {
	match act {
		.sigmoid { return activated_val * (1.0 - activated_val) }
		.relu { return if _likely_(activated_val > 0) { 1.0 } else { 0.0 } }
		.tanh { return 1.0 - (activated_val * activated_val) }
		.linear { return 1.0 }
	}
}

pub struct Layer {
pub mut:
	weights         Matrix
	biases          Matrix
	activation_type ActivationType
	last_input      Matrix
	last_output     Matrix
	delta           Matrix
	grad_w          Matrix
	m_w             Matrix
	v_w             Matrix
	m_b             Matrix
	v_b             Matrix
	beta1_t         f64
	beta2_t         f64
}

pub fn (mut l Layer) free() {
	l.weights.free()
	l.biases.free()
	l.last_input.free()
	l.last_output.free()
	l.delta.free()
	l.grad_w.free()
	l.m_w.free()
	l.v_w.free()
	l.m_b.free()
	l.v_b.free()
}

pub struct NeuralNetwork {
pub mut:
	layers    []Layer
	optimizer OptimizerType
	normalize bool = true
	means     []f64
	stds      []f64
}

pub fn (mut nn NeuralNetwork) free() {
	for mut layer in nn.layers {
		layer.free()
	}
	unsafe { 
		nn.layers.free() 
		if nn.means.len > 0 { nn.means.free() }
		if nn.stds.len > 0 { nn.stds.free() }
	}
}

pub struct Sequential {
pub mut:
	net NeuralNetwork
}

pub fn new_sequential(optimizer OptimizerType) Sequential {
	return Sequential{
		net: NeuralNetwork{
			optimizer: optimizer
			normalize: true
		}
	}
}

pub fn (mut s Sequential) add(input_size int, output_size int, act ActivationType) {
	s.net.layers << Layer{
		weights: new_random_matrix(output_size, input_size)
		biases: new_random_matrix(output_size, 1)
		activation_type: act
		last_input: new_matrix(input_size, 1)
		last_output: new_matrix(output_size, 1)
		delta: new_matrix(output_size, 1)
		grad_w: new_matrix(output_size, input_size)
		m_w: new_matrix(output_size, input_size)
		v_w: new_matrix(output_size, input_size)
		m_b: new_matrix(output_size, 1)
		v_b: new_matrix(output_size, 1)
		beta1_t: 1.0
		beta2_t: 1.0
	}
}

pub fn (mut s Sequential) set_normalize(val bool) {
	s.net.normalize = val
}

pub fn (mut s Sequential) free() {
	s.net.free()
}

pub fn (s &Sequential) save(path string) ! {
	data := json.encode(s.net)
	os.write_file(path, data)!
}

pub fn load_sequential(path string) !Sequential {
	data := os.read_file(path)!
	net := json.decode(NeuralNetwork, data)!
	return Sequential{
		net: net
	}
}

@[inline]
fn vnm_log(msg string) {
	$if vnm_safe ? {
		println("[VNM-SAFE] ${msg}")
	}
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) compute_normalization_params(inputs []Tensor) {
	if inputs.len == 0 { return }
	feat_size := inputs[0].data.len

	unsafe {
		if nn.means.len > 0 { nn.means.free() }
		if nn.stds.len > 0 { nn.stds.free() }
	}

	nn.means = []f64{len: feat_size, init: 0.0}
	nn.stds = []f64{len: feat_size, init: 0.0}

	unsafe {
		mut ptr_mean_start := &nn.means[0]
		mut ptr_std_start := &nn.stds[0]
		
		mut ptr_in := &inputs[0].data[0]
		mut ptr_m := ptr_mean_start
		mut ptr_s := ptr_std_start

		for i in 0 .. inputs.len {
			ptr_in = &inputs[i].data[0]
			ptr_m = ptr_mean_start
			for _ in 0 .. feat_size {
				*ptr_m += *ptr_in
				ptr_in++
				ptr_m++
			}
		}

		inv_m := 1.0 / f64(inputs.len)
		ptr_m = ptr_mean_start
		for _ in 0 .. feat_size {
			*ptr_m *= inv_m
			ptr_m++
		}

		for i in 0 .. inputs.len {
			ptr_in = &inputs[i].data[0]
			ptr_m = ptr_mean_start
			ptr_s = ptr_std_start
			for _ in 0 .. feat_size {
				diff := *ptr_in - *ptr_m
				*ptr_s += diff * diff
				ptr_in++
				ptr_m++
				ptr_s++
			}
		}

		ptr_s = ptr_std_start
		eps := 1e-8
		for _ in 0 .. feat_size {
			*ptr_s = math.sqrt(*ptr_s * inv_m) + eps
			ptr_s++
		}
	}
}

@[manualfree]
pub fn (mut s Sequential) predict(input Tensor) !Tensor {
	unsafe {
		return s.net.predict(input)!
	}
}

@[manualfree]
pub fn (mut s Sequential) train(inputs []Tensor, targets []Tensor, epochs int, lr f64) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, lr, 1.0, 0)!
	}
}

@[manualfree]
pub fn (mut s Sequential) train_with_decay(inputs []Tensor, targets []Tensor, epochs int, lr f64, decay_rate f64, decay_steps int) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, lr, decay_rate, decay_steps)!
	}
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) predict(input Tensor) !Tensor {
	return nn.predict_internal(input, true)!
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) predict_internal(input Tensor, perform_normalization bool) !Tensor {
	$if vnm_safe ? {
		if nn.layers.len == 0 {
			return error("Prediction failed: NeuralNetwork has no layers.")
		}
		first_layer_input_size := nn.layers[0].weights.cols
		if input.data.len != first_layer_input_size {
			return error("Dimension mismatch: Input size is ${input.data.len}, but network expects ${first_layer_input_size}.")
		}
		vnm_log("Input dimension verified: ${input.data.len}")
	}

	mut input_data := input.data

	if perform_normalization && nn.normalize && nn.means.len > 0 {
		input_data = input.data.clone()
		unsafe {
			mut ptr_data := &input_data[0]
			mut ptr_mean := &nn.means[0]
			mut ptr_std := &nn.stds[0]
			for _ in 0 .. input_data.len {
				*ptr_data = (*ptr_data - *ptr_mean) / *ptr_std
				ptr_data++
				ptr_mean++
				ptr_std++
			}
		}
	}

	mut current := Matrix{
		rows: input_data.len
		cols: 1
		data: input_data
	}

	for mut layer in nn.layers {
		layer.last_input = current
		mut raw_output := matmul_parallel(layer.weights, current)!
		
		unsafe {
			mut ptr_res := &raw_output.data[0]
			mut ptr_bias := &layer.biases.data[0]
			for _ in 0 .. raw_output.data.len {
				*ptr_res = activate(*ptr_res + *ptr_bias, layer.activation_type)
				ptr_res++
				ptr_bias++
			}
		}
		layer.last_output = raw_output
		current = raw_output
	}
	
	return Tensor{
		shape: [current.data.len]
		data: current.data
	}
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) train_step(input Tensor, target Tensor, lr f64) !Tensor {
	return nn.train_step_internal(input, target, lr, false)!
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) train_step_internal(input Tensor, target Tensor, lr f64, is_normalized bool) !Tensor {
	$if vnm_safe ? {
		if nn.layers.len == 0 {
			return error("Training failed: NeuralNetwork has no layers.")
		}
		last_layer_idx := nn.layers.len - 1
		expected_output_size := nn.layers[last_layer_idx].weights.rows
		if target.data.len != expected_output_size {
			return error("Dimension mismatch: Target size is ${target.data.len}, but network outputs ${expected_output_size}.")
		}
		vnm_log("Target dimension verified: ${target.data.len}")
	}

	output_tensor := nn.predict_internal(input, !is_normalized)!

	mut output_layer := &nn.layers[nn.layers.len - 1]
	mut delta := new_matrix(output_layer.last_output.rows, 1)
	
	unsafe {
		mut ptr_delta := &delta.data[0]
		mut ptr_out := &output_layer.last_output.data[0]
		mut ptr_target := &target.data[0]
		for _ in 0 .. delta.data.len {
			error_val := *ptr_out - *ptr_target
			*ptr_delta = error_val * activate_derivative(*ptr_out, output_layer.activation_type)
			ptr_delta++
			ptr_out++
			ptr_target++
		}
	}

	for l := nn.layers.len - 1; l >= 0; l-- {
		mut current_layer := &nn.layers[l]
		
		matmul_transpose_b_inplace(delta, current_layer.last_input, mut current_layer.grad_w)

		mut next_delta := new_matrix(1, 1)
		if _likely_(l > 0) {
			prev_layer := &nn.layers[l - 1]
			next_delta = new_matrix(prev_layer.last_output.rows, 1)
			
			matmul_transpose_a_inplace(current_layer.weights, delta, mut next_delta)
			
			unsafe {
				mut ptr_next := &next_delta.data[0]
				mut ptr_prev_out := &prev_layer.last_output.data[0]
				for _ in 0 .. next_delta.data.len {
					*ptr_next = *ptr_next * activate_derivative(*ptr_prev_out, prev_layer.activation_type)
					ptr_next++
					ptr_prev_out++
				}
			}
		}

		if _likely_(nn.optimizer == .adam) {
			beta1 := 0.9
			beta2 := 0.999
			eps := 1e-8

			current_layer.beta1_t *= beta1
			current_layer.beta2_t *= beta2

			bias_correction1 := 1.0 - current_layer.beta1_t
			bias_correction2 := 1.0 - current_layer.beta2_t

			unsafe {
				mut ptr_w := &current_layer.weights.data[0]
				mut ptr_g := &current_layer.grad_w.data[0]
				mut ptr_mw := &current_layer.m_w.data[0]
				mut ptr_vw := &current_layer.v_w.data[0]
				
				for _ in 0 .. current_layer.weights.data.len {
					*ptr_mw = beta1 * *ptr_mw + (1.0 - beta1) * *ptr_g
					*ptr_vw = beta2 * *ptr_vw + (1.0 - beta2) * *ptr_g * *ptr_g

					m_hat := *ptr_mw / bias_correction1
					v_hat := *ptr_vw / bias_correction2

					*ptr_w -= lr * m_hat / (math.sqrt(v_hat) + eps)
					ptr_w++
					ptr_g++
					ptr_mw++
					ptr_vw++
				}

				mut ptr_b := &current_layer.biases.data[0]
				mut ptr_d := &delta.data[0]
				mut ptr_mb := &current_layer.m_b.data[0]
				mut ptr_vb := &current_layer.v_b.data[0]

				for _ in 0 .. current_layer.biases.data.len {
					*ptr_mb = beta1 * *ptr_mb + (1.0 - beta1) * *ptr_d
					*ptr_vb = beta2 * *ptr_vb + (1.0 - beta2) * *ptr_d * *ptr_d

					m_hat := *ptr_mb / bias_correction1
					v_hat := *ptr_vb / bias_correction2

					*ptr_b -= lr * m_hat / (math.sqrt(v_hat) + eps)
					ptr_b++
					ptr_d++
					ptr_mb++
					ptr_vb++
				}
			}
		} else {
			unsafe {
				mut ptr_w := &current_layer.weights.data[0]
				mut ptr_g := &current_layer.grad_w.data[0]
				for _ in 0 .. current_layer.weights.data.len {
					*ptr_w -= lr * *ptr_g
					ptr_w++
					ptr_g++
				}
				
				mut ptr_b := &current_layer.biases.data[0]
				mut ptr_d := &delta.data[0]
				for _ in 0 .. current_layer.biases.data.len {
					*ptr_b -= lr * *ptr_d
					ptr_b++
					ptr_d++
				}
			}
		}

		delta.free()
		delta = next_delta
	}
	
	delta.free()
	return output_tensor
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) train_with_decay(inputs []Tensor, targets []Tensor, epochs int, lr f64, decay_rate f64, decay_steps int) ! {
	last_layer_idx := nn.layers.len - 1
	mut current_lr := lr

	mut training_inputs := inputs
	mut temp_normalized_tensors := []Tensor{}

	if nn.normalize {
		nn.compute_normalization_params(inputs)
		temp_normalized_tensors = []Tensor{cap: inputs.len}
		for i in 0 .. inputs.len {
			mut norm_data := inputs[i].data.clone()
			unsafe {
				mut ptr_data := &norm_data[0]
				mut ptr_mean := &nn.means[0]
				mut ptr_std := &nn.stds[0]
				for _ in 0 .. norm_data.len {
					*ptr_data = (*ptr_data - *ptr_mean) / *ptr_std
					ptr_data++
					ptr_mean++
					ptr_std++
				}
			}
			temp_normalized_tensors << Tensor{
				shape: inputs[i].shape.clone()
				data: norm_data
			}
		}
		training_inputs = temp_normalized_tensors
	}

	for epoch in 0 .. epochs {
		if _unlikely_(decay_rate < 1.0 && decay_steps > 0 && epoch > 0 && epoch % decay_steps == 0) {
			current_lr *= decay_rate
		}

		mut total_error := 0.0
		for i in 0 .. training_inputs.len {
			out := nn.train_step_internal(training_inputs[i], targets[i], current_lr, true)!
			unsafe {
				last_layer_output := nn.layers[last_layer_idx].last_output.data
				mut ptr_out := &last_layer_output[0]
				mut ptr_target := &targets[i].data[0]
				for _ in 0 .. last_layer_output.len {
					diff := *ptr_out - *ptr_target
					total_error += diff * diff
					ptr_out++
					ptr_target++
				}
			}
			out.free()
		}
		
		if epochs >= 10 && epoch % (epochs / 10) == 0 {
			mean_error := total_error / f64(inputs.len)
			println("  Epoch ${epoch:5d} / ${epochs} | Active LR: ${current_lr:.6f} | Mean Squared Loss: ${mean_error:.8f}")
		}
	}

	if nn.normalize {
		for mut t in temp_normalized_tensors {
			t.free()
		}
		unsafe { temp_normalized_tensors.free() }
	}
}
module vnm

import math
import rand
import os
import json
import runtime

#flag -O3
#flag -ffast-math
#flag -march=native
#flag -funroll-loops

fn C.memcpy(dest voidptr, src voidptr, n usize) voidptr

$if vnm_f64 ? {
	pub type Real = f64
} $else {
	pub type Real = f32
}

@[inline]
fn to_real_array(data []f64) []Real {
	mut res := []Real{len: data.len}
	unsafe {
		for i in 0 .. data.len {
			res[i] = Real(data[i])
		}
	}
	return res
}

$if vnm_f64 ? {
	@[inline]
	fn fast_exp(x Real) Real {
		if x < -700.0 { return Real(0.0) }
		if x > 700.0 { return Real(1.7976931348623157e+308) }
		
		val_f := 6497334751787128.0 * x + 4607063855013146400.0
		val_u := u64(val_f)
		mut res := Real(0.0)
		unsafe {
			C.memcpy(&res, &val_u, sizeof(Real))
		}
		return res
	}
	
	@[inline]
	fn fast_inv_sqrt(x Real) Real {
		mut xhalf := Real(0.5) * x
		mut i := u64(0)
		unsafe {
			C.memcpy(&i, &x, sizeof(Real))
		}
		i = 0x5fe6eb50c7b537a9 - (i >> 1)
		mut y := Real(0.0)
		unsafe {
			C.memcpy(&y, &i, sizeof(Real))
		}
		y = y * (Real(1.5) - xhalf * y * y)
		y = y * (Real(1.5) - xhalf * y * y)
		return y
	}

	@[inline]
	fn fast_sqrt(x Real) Real {
		return math.sqrt(x)
	}
} $else {
	fn C.sqrtf(x f32) f32

	@[inline]
	fn fast_exp(x Real) Real {
		if x < -88.0 { return Real(0.0) }
		if x > 88.0 { return Real(3.40282347e+38) }
		
		val_f := 12102203.0 * x + 1064866816.0
		val_u := u32(val_f)
		mut res := Real(0.0)
		unsafe {
			C.memcpy(&res, &val_u, sizeof(Real))
		}
		return res
	}
	
	@[inline]
	fn fast_inv_sqrt(x Real) Real {
		mut xhalf := Real(0.5) * x
		mut i := u32(0)
		unsafe {
			C.memcpy(&i, &x, sizeof(Real))
		}
		i = 0x5f3759df - (i >> 1)
		mut y := Real(0.0)
		unsafe {
			C.memcpy(&y, &i, sizeof(Real))
		}
		y = y * (Real(1.5) - xhalf * y * y)
		return y
	}

	@[inline]
	fn fast_sqrt(x Real) Real {
		return C.sqrtf(x)
	}
}

@[inline]
fn fast_sigmoid(x Real) Real {
	return Real(1.0) / (Real(1.0) + fast_exp(-x))
}

@[inline]
fn fast_tanh(x Real) Real {
	if x < Real(-5.0) { return Real(-1.0) }
	if x > Real(5.0) { return Real(1.0) }
	ex2 := fast_exp(Real(2.0) * x)
	return (ex2 - Real(1.0)) / (ex2 + Real(1.0))
}

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
	data []Real
}

pub fn new_tensor(shape []int, data []f64) Tensor {
	total_size := get_shape_size(shape)
	mut actual_data := to_real_array(data)
	if data.len < total_size {
		actual_data = []Real{len: total_size, init: Real(0.0)}
	}
	return Tensor{
		shape: shape
		data: actual_data
	}
}

pub fn vector(data []f64) Tensor {
	return Tensor{
		shape: [data.len, 1]
		data: to_real_array(data)
	}
}

pub fn scalar(val f64) Tensor {
	return Tensor{
		shape: [1, 1]
		data: [Real(val)]
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
	data []Real
}

pub fn new_matrix(rows int, cols int) Matrix {
	return Matrix{
		rows: rows
		cols: cols
		data: []Real{len: rows * cols, init: Real(0.0)}
	}
}

@[inline]
fn rand_range(min Real, max Real) Real {
	val := rand.f64_in_range(min, max) or { 0.0 }
	return Real(val)
}

pub fn new_random_matrix(rows int, cols int) Matrix {
	mut m := new_matrix(rows, cols)
	boundary := fast_sqrt(Real(6.0) / Real(cols))
	for i in 0 .. m.data.len {
		m.data[i] = rand_range(-boundary, boundary)
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

pub fn (m Matrix) clone() Matrix {
	return Matrix{
		rows: m.rows
		cols: m.cols
		data: m.data.clone()
	}
}

pub fn (m &Matrix) free() {
	unsafe { m.data.free() }
}

@[inline; unsafe]
fn copy_real(dest &Real, src &Real, len int) {
	unsafe {
		for i in 0 .. len {
			dest[i] = src[i]
		}
	}
}

@[inline; unsafe]
fn zero_real(dest &Real, len int) {
	unsafe {
		for i in 0 .. len {
			dest[i] = Real(0.0)
		}
	}
}

struct MatmulArgs {
	a         &Matrix
	b         &Matrix
	res       &Matrix
	start_row int
	end_row   int
}

fn matmul_worker(args MatmulArgs) {
	unsafe {
		cols_a := args.a.cols
		cols_b := args.b.cols
		for i in args.start_row .. args.end_row {
			offset_res := i * cols_b
			offset_a := i * cols_a
			for k in 0 .. cols_a {
				val_a := args.a.data[offset_a + k]
				offset_b := k * cols_b
				mut ptr_res := &args.res.data[offset_res]
				mut ptr_b := &args.b.data[offset_b]
				mut j := 0
				for j < cols_b - 3 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					j += 4
				}
				for j < cols_b {
					ptr_res[j] += val_a * ptr_b[j]
					j++
				}
			}
		}
	}
}

@[inline; unsafe]
fn matmul_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		if a.rows < 64 {
			matmul_serial_inplace(a, b, mut res)
			return
		}
		cols_b := b.cols
		if cols_b == 1 {
			matmul_serial_inplace(a, b, mut res)
			return
		}
		zero_real(&res.data[0], res.data.len)
		num_threads := runtime.nr_cpus()
		mut threads := []thread{}
		rows_per_thread := a.rows / num_threads
		for t in 0 .. num_threads {
			start := t * rows_per_thread
			mut end := (t + 1) * rows_per_thread
			if t == num_threads - 1 {
				end = a.rows
			}
			if start >= end {
				continue
			}
			threads << spawn matmul_worker(MatmulArgs{
				a: &a
				b: &b
				res: &res
				start_row: start
				end_row: end
			})
		}
		threads.wait()
	}
}

@[inline; unsafe]
fn matmul_serial_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		cols_a := a.cols
		cols_b := b.cols
		if cols_b == 1 {
			mut ptr_res := &res.data[0]
			mut ptr_a := &a.data[0]
			ptr_b_start := &b.data[0]
			for _ in 0 .. a.rows {
				mut sum := Real(0.0)
				mut ptr_b := ptr_b_start
				mut k := 0
				for k < cols_a - 3 {
					sum += ptr_a[k] * ptr_b[k]
					sum += ptr_a[k+1] * ptr_b[k+1]
					sum += ptr_a[k+2] * ptr_b[k+2]
					sum += ptr_a[k+3] * ptr_b[k+3]
					k += 4
				}
				for k < cols_a {
					sum += ptr_a[k] * ptr_b[k]
					k++
				}
				*ptr_res = sum
				ptr_res++
				ptr_a += cols_a
			}
			return
		}
		zero_real(&res.data[0], res.data.len)
		for i in 0 .. a.rows {
			offset_res := i * cols_b
			offset_a := i * cols_a
			for k in 0 .. cols_a {
				val_a := a.data[offset_a + k]
				offset_b := k * cols_b
				mut ptr_res := &res.data[offset_res]
				mut ptr_b := &res.data[offset_b]
				mut j := 0
				for j < cols_b - 3 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					j += 4
				}
				for j < cols_b {
					ptr_res[j] += val_a * ptr_b[j]
					j++
				}
			}
		}
	}
}

@[inline; unsafe]
fn matmul_transpose_b_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		mut ptr_res := &res.data[0]
		b_len := b.rows * b.cols
		for i in 0 .. a.rows {
			val_a := a.data[i]
			mut ptr_b := &b.data[0]
			mut j := 0
			for j < b_len - 3 {
				ptr_res[j] = val_a * ptr_b[j]
				ptr_res[j+1] = val_a * ptr_b[j+1]
				ptr_res[j+2] = val_a * ptr_b[j+2]
				ptr_res[j+3] = val_a * ptr_b[j+3]
				j += 4
			}
			for j < b_len {
				ptr_res[j] = val_a * ptr_b[j]
				j++
			}
			ptr_res += b_len
		}
	}
}

@[inline; unsafe]
fn matmul_transpose_a_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		zero_real(&res.data[0], res.data.len)
		cols_a := a.cols
		mut ptr_res := &res.data[0]
		for k in 0 .. a.rows {
			val_b := b.data[k]
			offset_a := k * cols_a
			mut ptr_a := &a.data[offset_a]
			mut i := 0
			for i < cols_a - 3 {
				ptr_res[i] += ptr_a[i] * val_b
				ptr_res[i+1] += ptr_a[i+1] * val_b
				ptr_res[i+2] += ptr_a[i+2] * val_b
				ptr_res[i+3] += ptr_a[i+3] * val_b
				i += 4
			}
			for i < cols_a {
				ptr_res[i] += ptr_a[i] * val_b
				i++
			}
		}
	}
}

@[inline; unsafe]
fn matmul_add_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		cols_a := a.cols
		cols_b := b.cols
		if cols_b == 1 {
			mut ptr_res := &res.data[0]
			mut ptr_a := &a.data[0]
			ptr_b_start := &b.data[0]
			for _ in 0 .. a.rows {
				mut sum := Real(0.0)
				mut ptr_b := ptr_b_start
				mut k := 0
				for k < cols_a - 3 {
					sum += ptr_a[k] * ptr_b[k]
					sum += ptr_a[k+1] * ptr_b[k+1]
					sum += ptr_a[k+2] * ptr_b[k+2]
					sum += ptr_a[k+3] * ptr_b[k+3]
					k += 4
				}
				for k < cols_a {
					sum += ptr_a[k] * ptr_b[k]
					k++
				}
				*ptr_res += sum
				ptr_res++
				ptr_a += cols_a
			}
			return
		}
		for i in 0 .. a.rows {
			offset_res := i * cols_b
			offset_a := i * cols_a
			for k in 0 .. cols_a {
				val_a := a.data[offset_a + k]
				offset_b := k * cols_b
				mut ptr_res := &res.data[offset_res]
				mut ptr_b := &b.data[offset_b]
				mut j := 0
				for j < cols_b - 3 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					j += 4
				}
				for j < cols_b {
					ptr_res[j] += val_a * ptr_b[j]
					j++
				}
			}
		}
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
	beta1_t         Real
	beta2_t         Real
	dropout_rate    Real
	is_rnn          bool
	hidden_weights  Matrix
	prev_hidden     Matrix
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
	l.hidden_weights.free()
	l.prev_hidden.free()
}

pub struct NeuralNetwork {
pub mut:
	layers    []Layer
	optimizer OptimizerType
	normalize bool = true
	means     []Real
	stds      []Real
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

pub fn (mut s Sequential) add(input_size int, output_size int, act ActivationType, dropout_rate Real, is_rnn bool) {
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
		beta1_t: Real(1.0)
		beta2_t: Real(1.0)
		dropout_rate: dropout_rate
		is_rnn: is_rnn
		hidden_weights: if is_rnn { new_random_matrix(output_size, output_size) } else { new_matrix(1, 1) }
		prev_hidden: if is_rnn { new_matrix(output_size, 1) } else { new_matrix(1, 1) }
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

	nn.means = []Real{len: feat_size, init: Real(0.0)}
	nn.stds = []Real{len: feat_size, init: Real(0.0)}

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

		inv_m := Real(1.0) / Real(inputs.len)
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
		eps := Real(1e-8)
		for _ in 0 .. feat_size {
			*ptr_s = fast_sqrt(*ptr_s * inv_m) + eps
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
pub fn (mut s Sequential) train(inputs []Tensor, targets []Tensor, epochs int, lr Real) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, lr, Real(1.0), 0)!
	}
}

@[manualfree]
pub fn (mut s Sequential) train_with_decay(inputs []Tensor, targets []Tensor, epochs int, lr Real, decay_rate Real, decay_steps int) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, lr, decay_rate, decay_steps)!
	}
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) predict(input Tensor) !Tensor {
	return nn.predict_internal(input, true)!
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) forward_pass(input Tensor, perform_normalization bool) ! {
	$if vnm_safe ? {
		if nn.layers.len == 0 {
			return error("Forward pass failed: NeuralNetwork has no layers.")
		}
		first_layer_input_size := nn.layers[0].weights.cols
		if input.data.len != first_layer_input_size {
			return error("Dimension mismatch: Input size is ${input.data.len}, but network expects ${first_layer_input_size}.")
		}
	}

	unsafe {
		mut first_layer := &nn.layers[0]

		if perform_normalization && nn.normalize && nn.means.len > 0 {
			mut ptr_dest := &first_layer.last_input.data[0]
			mut ptr_src := &input.data[0]
			mut ptr_mean := &nn.means[0]
			mut ptr_std := &nn.stds[0]
			for _ in 0 .. input.data.len {
				*ptr_dest = (*ptr_src - *ptr_mean) / *ptr_std
				ptr_dest++
				ptr_src++
				ptr_mean++
				ptr_std++
			}
		} else {
			copy_real(&first_layer.last_input.data[0], &input.data[0], input.data.len)
		}

		for l in 0 .. nn.layers.len {
			mut layer := &nn.layers[l]
			
			if l > 0 {
				prev_layer := &nn.layers[l - 1]
				copy_real(&layer.last_input.data[0], &prev_layer.last_output.data[0], layer.last_input.data.len)
			}
			
			matmul_inplace(layer.weights, layer.last_input, mut layer.last_output)
			
			if layer.is_rnn {
				matmul_add_inplace(layer.hidden_weights, layer.prev_hidden, mut layer.last_output)
			}
			
			mut ptr_res := &layer.last_output.data[0]
			mut ptr_bias := &layer.biases.data[0]
			len := layer.last_output.data.len
			
			match layer.activation_type {
				.sigmoid {
					for i in 0 .. len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = fast_sigmoid(val)
					}
				}
				.relu {
					for i in 0 .. len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = if val > Real(0.0) { val } else { Real(0.0) }
					}
				}
				.tanh {
					for i in 0 .. len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = fast_tanh(val)
					}
				}
				.linear {
					for i in 0 .. len {
						ptr_res[i] = ptr_res[i] + ptr_bias[i]
					}
				}
			}

			if layer.is_rnn {
				copy_real(&layer.prev_hidden.data[0], &layer.last_output.data[0], layer.last_output.data.len)
			}
		}
	}
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) predict_internal(input Tensor, perform_normalization bool) !Tensor {
	unsafe {
		nn.forward_pass(input, perform_normalization)!
		last_layer := &nn.layers[nn.layers.len - 1]
		return Tensor{
			shape: [last_layer.last_output.data.len]
			data: last_layer.last_output.data.clone()
		}
	}
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) train_step_internal(input Tensor, target Tensor, lr Real, is_normalized bool) !Real {
	$if vnm_safe ? {
		if nn.layers.len == 0 {
			return error("Training failed: NeuralNetwork has no layers.")
		}
		last_layer_idx := nn.layers.len - 1
		expected_output_size := nn.layers[last_layer_idx].weights.rows
		if target.data.len != expected_output_size {
			return error("Dimension mismatch: Target size is ${target.data.len}, but network outputs ${expected_output_size}.")
		}
	}

	unsafe {
		nn.forward_pass(input, !is_normalized)!

		mut output_layer := &nn.layers[nn.layers.len - 1]
		mut step_loss := Real(0.0)
		
		mut ptr_delta := &output_layer.delta.data[0]
		mut ptr_out := &output_layer.last_output.data[0]
		mut ptr_target := &target.data[0]
		len := output_layer.delta.data.len
		
		match output_layer.activation_type {
			.sigmoid {
				for i in 0 .. len {
					error_val := ptr_out[i] - ptr_target[i]
					step_loss += error_val * error_val
					ptr_delta[i] = error_val * (ptr_out[i] * (Real(1.0) - ptr_out[i]))
				}
			}
			.relu {
				for i in 0 .. len {
					error_val := ptr_out[i] - ptr_target[i]
					step_loss += error_val * error_val
					ptr_delta[i] = if ptr_out[i] > Real(0.0) { error_val } else { Real(0.0) }
				}
			}
			.tanh {
				for i in 0 .. len {
					error_val := ptr_out[i] - ptr_target[i]
					step_loss += error_val * error_val
					ptr_delta[i] = error_val * (Real(1.0) - (ptr_out[i] * ptr_out[i]))
				}
			}
			.linear {
				for i in 0 .. len {
					error_val := ptr_out[i] - ptr_target[i]
					step_loss += error_val * error_val
					ptr_delta[i] = error_val
				}
			}
		}

		for l := nn.layers.len - 1; l >= 0; l-- {
			mut current_layer := &nn.layers[l]
			
			matmul_transpose_b_inplace(current_layer.delta, current_layer.last_input, mut current_layer.grad_w)

			if l > 0 {
				mut prev_layer := &nn.layers[l - 1]
				
				matmul_transpose_a_inplace(current_layer.weights, current_layer.delta, mut prev_layer.delta)
				
				mut ptr_next := &prev_layer.delta.data[0]
				mut ptr_prev_out := &prev_layer.last_output.data[0]
				len_prev := prev_layer.delta.data.len
				
				match prev_layer.activation_type {
					.sigmoid {
						for i in 0 .. len_prev {
							ptr_next[i] = ptr_next[i] * (ptr_prev_out[i] * (Real(1.0) - ptr_prev_out[i]))
						}
					}
					.relu {
						for i in 0 .. len_prev {
							ptr_next[i] = if ptr_prev_out[i] > Real(0.0) { ptr_next[i] } else { Real(0.0) }
						}
					}
					.tanh {
						for i in 0 .. len_prev {
							ptr_next[i] = ptr_next[i] * (Real(1.0) - (ptr_prev_out[i] * ptr_prev_out[i]))
						}
					}
					.linear {}
				}
			}

			if _likely_(nn.optimizer == .adam) {
				beta1 := Real(0.9)
				beta2 := Real(0.999)
				
				eps_sq := Real(1e-12)

				current_layer.beta1_t *= beta1
				current_layer.beta2_t *= beta2

				bias_correction1 := Real(1.0) - current_layer.beta1_t
				bias_correction2 := Real(1.0) - current_layer.beta2_t

				mut ptr_w := &current_layer.weights.data[0]
				mut ptr_g := &current_layer.grad_w.data[0]
				mut ptr_mw := &current_layer.m_w.data[0]
				mut ptr_vw := &current_layer.v_w.data[0]
				
				one_minus_beta1 := Real(1.0) - beta1
				one_minus_beta2 := Real(1.0) - beta2
				
				step_size := lr / bias_correction1
				inv_bias_corr2 := Real(1.0) / bias_correction2
				
				len_w := current_layer.weights.data.len
				
				for i in 0 .. len_w {
					mw_val := beta1 * ptr_mw[i] + one_minus_beta1 * ptr_g[i]
					vw_val := beta2 * ptr_vw[i] + one_minus_beta2 * ptr_g[i] * ptr_g[i]
					
					ptr_mw[i] = mw_val
					ptr_vw[i] = vw_val
					
					v_hat := vw_val * inv_bias_corr2
					ptr_w[i] -= step_size * mw_val * fast_inv_sqrt(v_hat + eps_sq)
				}

				mut ptr_b := &current_layer.biases.data[0]
				mut ptr_d := &current_layer.delta.data[0]
				mut ptr_mb := &current_layer.m_b.data[0]
				mut ptr_vb := &current_layer.v_b.data[0]

				len_b := current_layer.biases.data.len
				for i in 0 .. len_b {
					mb_val := beta1 * ptr_mb[i] + one_minus_beta1 * ptr_d[i]
					vb_val := beta2 * ptr_vb[i] + one_minus_beta2 * ptr_d[i] * ptr_d[i]
					
					ptr_mb[i] = mb_val
					ptr_vb[i] = vb_val
					
					v_hat_b := vb_val * inv_bias_corr2
					ptr_b[i] -= step_size * mb_val * fast_inv_sqrt(v_hat_b + eps_sq)
				}
			} else {
				mut ptr_w := &current_layer.weights.data[0]
				mut ptr_g := &current_layer.grad_w.data[0]
				len_w := current_layer.weights.data.len
				for i in 0 .. len_w {
					ptr_w[i] -= lr * ptr_g[i]
				}
				
				mut ptr_b := &current_layer.biases.data[0]
				mut ptr_d := &current_layer.delta.data[0]
				len_b := current_layer.biases.data.len
				for i in 0 .. len_b {
					ptr_b[i] -= lr * ptr_d[i]
				}
			}
		}
		return step_loss
	}
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) train_with_decay(inputs []Tensor, targets []Tensor, epochs int, lr Real, decay_rate Real, decay_steps int) ! {
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
		if _unlikely_(decay_rate < Real(1.0) && decay_steps > 0 && epoch > 0 && epoch % decay_steps == 0) {
			current_lr *= decay_rate
		}

		mut total_error := Real(0.0)
		for i in 0 .. training_inputs.len {
			step_loss := nn.train_step_internal(training_inputs[i], targets[i], current_lr, true)!
			total_error += step_loss
		}
		
		if epochs >= 10 && epoch % (epochs / 10) == 0 {
			mean_error := total_error / Real(inputs.len)
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

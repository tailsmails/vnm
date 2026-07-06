module vnm

import math
import rand
import os
import runtime

#flag -Ofast
#flag -ffast-math
#flag -march=native
#flag -funroll-loops
#flag -flto
#flag -fomit-frame-pointer
#flag -ftree-vectorize
#flag -falign-loops=32
#flag -I @VMODROOT

$if vnm_f16 ? || ((arm64 || aarch64) && !vnm_f32 ? && !vnm_f64 ?) {
	#flag -DVNM_F16
}

#include "vnm_arm64.c"

$if (arm64 || aarch64) && !vnm_f64 ? {
	fn C.neon_dot_product_arm64(a &Real, b &Real, len int) f32
	fn C.approx_inv_sqrt_neon(x f32) f32
	fn C.approx_sigmoid_neon(x f32) f32
	fn C.approx_tanh_neon(x f32) f32
}

fn C.memcpy(dest voidptr, src voidptr, n usize) voidptr

const size_of_int = 4
const size_of_f64 = 8

$if vnm_f64 ? {
	pub type Real = f64
	pub type Fnn = f64
	const size_of_real = 8
	@[inline] pub fn to_real(val f64) Real { return Real(val) }
} $else {
	pub type Real = f32
	pub type Fnn = f32
	const size_of_real = 4
	@[inline] pub fn to_real(val f64) Real { return Real(val) }
}

pub type ActivationFn = fn (Real) Real
pub type DerivativeFn = fn (Real) Real

fn dummy_act(x Real) Real {
	return x
}

pub struct CustomActivation {
pub:
	forward    ActivationFn = dummy_act
	derivative DerivativeFn = dummy_act
}

struct DummyJsonStruct {
	v int
}

const block_size = 64

fn init() {
	$if (vnm_f16 ? || ((arm64 || aarch64) && !vnm_f32 ? && !vnm_f64 ?)) && !arm64 && !aarch64 {
		println('[VNM] Warning: f16 support is only available on arm64 (AArch64) architectures!')
		exit(1)
	}
}

@[inline]
pub fn (t Tensor) get(idx int) f64 {
	return f64(t.data[idx])
}

@[inline]
pub fn (m Matrix) get(r int, c int) f64 {
	return f64(m.data[r * m.cols + c])
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
	fn fast_sqrt(x Real) Real {
		return math.sqrt(x)
	}
} $else {
	fn C.sqrtf(x f32) f32

	@[inline]
	fn fast_sqrt(x Real) Real {
		return C.sqrtf(x)
	}
}

fn C.fmaxf(x f32, y f32) f32
fn C.fmax(x f64, y f64) f64

@[inline]
fn fast_max(a Real, b Real) Real {
	$if vnm_f64 ? {
		return C.fmax(a, b)
	} $else {
		return C.fmaxf(a, b)
	}
}

@[inline]
fn fast_exp(x Real) Real {
	$if vnm_f64 ? {
		mut cl_x := if x < to_real(-700.0) { to_real(-700.0) } else { x }
		cl_x = if cl_x > to_real(700.0) { to_real(700.0) } else { cl_x }
		fb := cl_x * to_real(6497320494453281.7)
		mut bits := u64(0)
		bits = u64(i64(fb) + 4607182418800017408)
		mut res := Real(0.0)
		unsafe { res = *(&Real(voidptr(&bits))) }
		return res
	} $else {
		mut cl_x := if x < to_real(-88.0) { to_real(-88.0) } else { x }
		cl_x = if cl_x > to_real(88.0) { to_real(88.0) } else { cl_x }
		fb := cl_x * to_real(12102203.0)
		mut bits := u32(0)
		bits = u32(int(fb) + 1065353216)
		mut res := Real(0.0)
		unsafe { res = *(&Real(voidptr(&bits))) }
		return res
	}
}

@[inline]
fn approx_sigmoid(x Real) Real {
	$if (arm64 || aarch64) && !vnm_f64 ? {
		return C.approx_sigmoid_neon(x)
	}
	abs_x := if x < to_real(0.0) { -x } else { x }
	return to_real(0.5) + (to_real(0.5) * x) / (to_real(1.0) + abs_x)
}

@[inline]
fn approx_tanh(x Real) Real {
	$if (arm64 || aarch64) && !vnm_f64 ? {
		return C.approx_tanh_neon(x)
	}
	abs_x := if x < to_real(0.0) { -x } else { x }
	return x / (to_real(1.0) + abs_x)
}

@[inline]
fn approx_inv_sqrt(x Real) Real {
	$if vnm_f64 ? {
		mut xhalf := to_real(0.5) * x
		mut i := u64(0)
		unsafe { i = *(&u64(voidptr(&x))) }
		i = 0x5fe6eb50c7b537a9 - (i >> 1)
		mut y := to_real(0.0)
		unsafe { y = *(&Real(voidptr(&i))) }
		y = y * (to_real(1.5) - xhalf * y * y)
		return y
	} $else {
		$if arm64 || aarch64 {
			return C.approx_inv_sqrt_neon(x)
		}
		mut xhalf := to_real(0.5) * x
		mut i := u32(0)
		unsafe { i = *(&u32(voidptr(&x))) }
		i = 0x5f3759df - (i >> 1)
		mut y := to_real(0.0)
		unsafe { y = *(&Real(voidptr(&i))) }
		y = y * (to_real(1.5) - xhalf * y * y)
		return y
	}
}

@[inline]
pub fn approx_gelu(x Real) Real {
	const_1 := to_real(0.79788456)
	const_2 := to_real(0.0356774)
	half := to_real(0.5)
	one := to_real(1.0)
	x_sq := x * x
	inner := x * (const_1 + const_2 * x_sq)
	return half * x * (one + approx_tanh(inner))
}

@[inline]
pub fn approx_log2(x Real) Real {
	$if vnm_f64 ? {
		mut bits := u64(0)
		unsafe { bits = *(&u64(voidptr(&x))) }
		mut val := Real(bits) - 4607182418800017408.0
		val *= 2.220446049250313e-16
		return val
	} $else {
		mut bits := u32(0)
		unsafe { bits = *(&u32(voidptr(&x))) }
		mut val := Real(bits) - 1065353216.0
		val *= 1.1920928955078125e-7
		return val
	}
}

@[inline]
pub fn approx_ln(x Real) Real {
	return approx_log2(x) * to_real(0.69314718056)
}

@[inline; unsafe]
pub fn softmax_row_inplace(mut data &Real, len int) {
	if len == 0 {
		return
	}
	unsafe {
		mut max_val := data[0]
		mut i := 1
		for i < len - 3 {
			v0 := data[i]
			v1 := data[i+1]
			v2 := data[i+2]
			v3 := data[i+3]
			if v0 > max_val { max_val = v0 }
			if v1 > max_val { max_val = v1 }
			if v2 > max_val { max_val = v2 }
			if v3 > max_val { max_val = v3 }
			i += 4
		}
		for i < len {
			if data[i] > max_val {
				max_val = data[i]
			}
			i++
		}
		
		mut sum := to_real(0.0)
		mut j := 0
		for j < len - 3 {
			data[j] = fast_exp(data[j] - max_val)
			data[j+1] = fast_exp(data[j+1] - max_val)
			data[j+2] = fast_exp(data[j+2] - max_val)
			data[j+3] = fast_exp(data[j+3] - max_val)
			sum += data[j] + data[j+1] + data[j+2] + data[j+3]
			j += 4
		}
		for j < len {
			data[j] = fast_exp(data[j] - max_val)
			sum += data[j]
			j++
		}
		
		inv_sum := to_real(1.0) / sum
		mut k := 0
		for k < len - 3 {
			data[k] *= inv_sum
			data[k+1] *= inv_sum
			data[k+2] *= inv_sum
			data[k+3] *= inv_sum
			k += 4
		}
		for k < len {
			data[k] *= inv_sum
			k++
		}
	}
}

@[inline; unsafe]
pub fn softmax_matrix_rows_inplace(mut m Matrix) {
	unsafe {
		for i in 0 .. m.rows {
			offset := i * m.cols
			softmax_row_inplace(mut &m.data[offset], m.cols)
		}
	}
}

pub enum ActivationType {
	sigmoid
	relu
	tanh
	linear
	custom
}

pub enum OptimizerType {
	sgd
	adam
}

pub enum QuantizationType {
	none
	int8
	int16
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
		actual_data = []Real{len: total_size, init: to_real(0.0)}
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
		data: [to_real(val)]
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
		data: []Real{len: rows * cols, init: to_real(0.0)}
	}
}

@[inline]
fn rand_range(min Real, max Real) Real {
	val := rand.f64_in_range(min, max) or { 0.0 }
	return Real(val)
}

pub fn new_random_matrix(rows int, cols int) Matrix {
	mut m := new_matrix(rows, cols)
	boundary := fast_sqrt(to_real(6.0) / to_real(cols))
	for i in 0 .. m.data.len {
		m.data[i] = rand_range(to_real(0.0) - boundary, boundary)
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
			dest[i] = to_real(0.0)
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
				for j < cols_b - 7 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					ptr_res[j+4] += val_a * ptr_b[j+4]
					ptr_res[j+5] += val_a * ptr_b[j+5]
					ptr_res[j+6] += val_a * ptr_b[j+6]
					ptr_res[j+7] += val_a * ptr_b[j+7]
					j += 8
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
		if a.rows < 256 {
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
fn matmul_tiled_inplace(a Matrix, b Matrix, mut res Matrix) {
	unsafe {
		zero_real(&res.data[0], res.data.len)
		m := a.rows
		k_sz := a.cols
		n := b.cols
		mut ih := 0
		for ih < m {
			i_end := if ih + block_size < m { ih + block_size } else { m }
			mut kh := 0
			for kh < k_sz {
				k_end := if kh + block_size < k_sz { kh + block_size } else { k_sz }
				mut jh := 0
				for jh < n {
					j_end := if jh + block_size < n { jh + block_size } else { n }
					for i in ih .. i_end {
						offset_res := i * n
						offset_a := i * k_sz
						for k in kh .. k_end {
							val_a := a.data[offset_a + k]
							offset_b := k * n
							mut ptr_res := &res.data[offset_res]
							mut ptr_b := &b.data[offset_b]
							for j in jh .. j_end {
								ptr_res[j] += val_a * ptr_b[j]
							}
						}
					}
					jh += block_size
				}
				kh += block_size
			}
			ih += block_size
		}
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
			$if (arm64 || aarch64) && !vnm_f64 ? {
				for _ in 0 .. a.rows {
					*ptr_res = to_real(C.neon_dot_product_arm64(ptr_a, ptr_b_start, cols_a))
					ptr_res++
					ptr_a += cols_a
				}
			} $else {
				for _ in 0 .. a.rows {
					mut sum0 := to_real(0.0)
					mut sum1 := to_real(0.0)
					mut sum2 := to_real(0.0)
					mut sum3 := to_real(0.0)
					mut sum4 := to_real(0.0)
					mut sum5 := to_real(0.0)
					mut sum6 := to_real(0.0)
					mut sum7 := to_real(0.0)
					mut ptr_b := ptr_b_start
					mut k := 0
					for k < cols_a - 7 {
						sum0 += ptr_a[k] * ptr_b[k]
						sum1 += ptr_a[k+1] * ptr_b[k+1]
						sum2 += ptr_a[k+2] * ptr_b[k+2]
						sum3 += ptr_a[k+3] * ptr_b[k+3]
						sum4 += ptr_a[k+4] * ptr_b[k+4]
						sum5 += ptr_a[k+5] * ptr_b[k+5]
						sum6 += ptr_a[k+6] * ptr_b[k+6]
						sum7 += ptr_a[k+7] * ptr_b[k+7]
						k += 8
					}
					for k < cols_a {
						sum0 += ptr_a[k] * ptr_b[k]
						k++
					}
					*ptr_res = (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7)
					ptr_res++
					ptr_a += cols_a
				}
			}
			return
		}
		if a.rows >= 128 || cols_a >= 128 || cols_b >= 128 {
			matmul_tiled_inplace(a, b, mut res)
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
				mut ptr_b := &b.data[offset_b]
				mut j := 0
				for j < cols_b - 7 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					ptr_res[j+4] += val_a * ptr_b[j+4]
					ptr_res[j+5] += val_a * ptr_b[j+5]
					ptr_res[j+6] += val_a * ptr_b[j+6]
					ptr_res[j+7] += val_a * ptr_b[j+7]
					j += 8
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
			for j in 0 .. b_len {
				ptr_res[j] = val_a * ptr_b[j]
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
			for i in 0 .. cols_a {
				ptr_res[i] += ptr_a[i] * val_b
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
			$if (arm64 || aarch64) && !vnm_f64 ? {
				for _ in 0 .. a.rows {
					*ptr_res += to_real(C.neon_dot_product_arm64(ptr_a, ptr_b_start, cols_a))
					ptr_res++
					ptr_a += cols_a
				}
			} $else {
				for _ in 0 .. a.rows {
					mut sum0 := to_real(0.0)
					mut sum1 := to_real(0.0)
					mut sum2 := to_real(0.0)
					mut sum3 := to_real(0.0)
					mut sum4 := to_real(0.0)
					mut sum5 := to_real(0.0)
					mut sum6 := to_real(0.0)
					mut sum7 := to_real(0.0)
					mut ptr_b := ptr_b_start
					mut k := 0
					for k < cols_a - 7 {
						sum0 += ptr_a[k] * ptr_b[k]
						sum1 += ptr_a[k+1] * ptr_b[k+1]
						sum2 += ptr_a[k+2] * ptr_b[k+2]
						sum3 += ptr_a[k+3] * ptr_b[k+3]
						sum4 += ptr_a[k+4] * ptr_b[k+4]
						sum5 += ptr_a[k+5] * ptr_b[k+5]
						sum6 += ptr_a[k+6] * ptr_b[k+6]
						sum7 += ptr_a[k+7] * ptr_b[k+7]
						k += 8
					}
					for k < cols_a {
						sum0 += ptr_a[k] * ptr_b[k]
						k++
					}
					*ptr_res += (sum0 + sum1 + sum2 + sum3) + (sum4 + sum5 + sum6 + sum7)
					ptr_res++
					ptr_a += cols_a
				}
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
				for j < cols_b - 7 {
					ptr_res[j] += val_a * ptr_b[j]
					ptr_res[j+1] += val_a * ptr_b[j+1]
					ptr_res[j+2] += val_a * ptr_b[j+2]
					ptr_res[j+3] += val_a * ptr_b[j+3]
					ptr_res[j+4] += val_a * ptr_b[j+4]
					ptr_res[j+5] += val_a * ptr_b[j+5]
					ptr_res[j+6] += val_a * ptr_b[j+6]
					ptr_res[j+7] += val_a * ptr_b[j+7]
					j += 8
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
pub fn matmul_qk_t_inplace(q Matrix, k Matrix, mut res Matrix, scale Real) {
	unsafe {
		cols_q := q.cols
		for i in 0 .. q.rows {
			offset_res := i * k.rows
			offset_q := i * cols_q
			for j in 0 .. k.rows {
				offset_k := j * cols_q
				mut sum := to_real(0.0)
				$if (arm64 || aarch64) && !vnm_f64 ? {
					sum = to_real(C.neon_dot_product_arm64(&q.data[offset_q], &k.data[offset_k], cols_q))
				} $else {
					for d in 0 .. cols_q {
						sum += q.data[offset_q + d] * k.data[offset_k + d]
					}
				}
				res.data[offset_res + j] = sum * scale
			}
		}
	}
}

@[inline; unsafe]
pub fn attention_forward_step(q Matrix, k Matrix, v Matrix, mut scores Matrix, mut output Matrix, scale Real) {
	unsafe {
		matmul_qk_t_inplace(q, k, mut scores, scale)
		softmax_matrix_rows_inplace(mut scores)
		matmul_inplace(scores, v, mut output)
	}
}

pub type CustomLayerForwardFn = fn (mut l Layer, input Matrix) Matrix
pub type CustomLayerBackwardFn = fn (mut l Layer, next_delta Matrix) Matrix

fn dummy_layer_forward(mut l Layer, input Matrix) Matrix {
	_ = input
	return l.last_output
}

fn dummy_layer_backward(mut l Layer, next_delta Matrix) Matrix {
	_ = next_delta
	return l.delta
}

pub struct Layer {
pub mut:
	weights         Matrix
	biases          Matrix
	activation_type ActivationType
	custom_act      CustomActivation
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
	is_custom       bool
	custom_forward  CustomLayerForwardFn = dummy_layer_forward
	custom_backward CustomLayerBackwardFn = dummy_layer_backward
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

pub type LossComputeFn = fn (output Matrix, target Matrix) Real
pub type LossGradientFn = fn (output Matrix, target Matrix, mut grad Matrix)

fn dummy_loss_compute(output Matrix, target Matrix) Real {
	_ = output
	_ = target
	return to_real(0.0)
}

fn dummy_loss_gradient(output Matrix, target Matrix, mut grad Matrix) {
	_ = output
	_ = target
	_ = grad
}

pub struct LossFunction {
pub:
	compute  LossComputeFn = dummy_loss_compute
	gradient LossGradientFn = dummy_loss_gradient
}

fn mse_compute(output Matrix, target Matrix) Real {
	mut sum := to_real(0.0)
	for i in 0 .. output.data.len {
		diff := output.data[i] - target.data[i]
		sum += diff * diff
	}
	return sum
}

fn mse_gradient(output Matrix, target Matrix, mut grad Matrix) {
	for i in 0 .. output.data.len {
		grad.data[i] = output.data[i] - target.data[i]
	}
}

pub const loss_mse = LossFunction{
	compute: mse_compute
	gradient: mse_gradient
}

pub struct NeuralNetwork {
pub mut:
	layers    []Layer
	optimizer OptimizerType
	normalize bool = true
	means     []Real
	stds      []Real
	loss_fn   LossFunction = loss_mse
}

pub fn (mut nn NeuralNetwork) free() {
	for mut layer in nn.layers {
		layer.free()
	}
	unsafe {
		nn.layers.free()
		if nn.means.len > 0 {
			nn.means.free()
		}
		if nn.stds.len > 0 {
			nn.stds.free()
		}
	}
}

struct ByteBuffer {
mut:
	data []u8
	pos  int
}

fn new_byte_buffer(cap int) ByteBuffer {
	return ByteBuffer{
		data: []u8{len: 0, cap: cap}
		pos: 0
	}
}

fn (mut b ByteBuffer) write_u8(val u8) {
	b.data << val
}

fn (mut b ByteBuffer) write_bool(val bool) {
	b.data << if val { u8(1) } else { u8(0) }
}

fn (mut b ByteBuffer) write_int(val int) {
	unsafe {
		ptr := &val
		b.write_bytes(ptr, size_of_int)
	}
}

fn (mut b ByteBuffer) write_f64(val f64) {
	unsafe {
		ptr := &val
		b.write_bytes(ptr, size_of_f64)
	}
}

fn (mut b ByteBuffer) write_real(val Real) {
	unsafe {
		ptr := &val
		b.write_bytes(ptr, size_of_real)
	}
}

fn (mut b ByteBuffer) write_bytes(ptr voidptr, len int) {
	unsafe {
		u8_ptr := &u8(ptr)
		for i in 0 .. len {
			b.data << u8_ptr[i]
		}
	}
}

fn (mut b ByteBuffer) write_matrix(m Matrix) {
	b.write_int(m.rows)
	b.write_int(m.cols)
	unsafe {
		if m.data.len > 0 {
			b.write_bytes(&m.data[0], m.data.len * size_of_real)
		}
	}
}

struct ByteReader {
	data []u8
mut:
	pos  int
}

fn (mut r ByteReader) read_u8() u8 {
	val := r.data[r.pos]
	r.pos++
	return val
}

fn (mut r ByteReader) read_bool() bool {
	return r.read_u8() == 1
}

fn (mut r ByteReader) read_int() int {
	unsafe {
		mut val := 0
		ptr := &u8(&val)
		for i in 0 .. size_of_int {
			ptr[i] = r.data[r.pos + i]
		}
		r.pos += size_of_int
		return val
	}
}

fn (mut r ByteReader) read_f64() f64 {
	unsafe {
		mut val := 0.0
		ptr := &u8(&val)
		for i in 0 .. size_of_f64 {
			ptr[i] = r.data[r.pos + i]
		}
		r.pos += size_of_f64
		return val
	}
}

fn (mut r ByteReader) read_real() Real {
	unsafe {
		mut val := Real(0.0)
		ptr := &u8(&val)
		for i in 0 .. size_of_real {
			ptr[i] = r.data[r.pos + i]
		}
		r.pos += size_of_real
		return val
	}
}

fn (mut r ByteReader) read_matrix() Matrix {
	rows := r.read_int()
	cols := r.read_int()
	mut m := new_matrix(rows, cols)
	unsafe {
		if m.data.len > 0 {
			ptr := &u8(&m.data[0])
			len_bytes := m.data.len * size_of_real
			for i in 0 .. len_bytes {
				ptr[i] = r.data[r.pos + i]
			}
			r.pos += len_bytes
		}
	}
	return m
}

pub struct Sequential {
pub mut:
	net          NeuralNetwork
	quantization QuantizationType = .none
}

pub fn new_sequential(optimizer OptimizerType) Sequential {
	return Sequential{
		net: NeuralNetwork{
			optimizer: optimizer
			normalize: true
		}
	}
}

pub fn (mut s Sequential) set_loss_function(loss LossFunction) {
	s.net.loss_fn = loss
}

pub fn (mut s Sequential) add(input_size int, output_size int, act ActivationType, dropout_rate f64, is_rnn bool) {
	s.net.layers << Layer{
		weights: new_random_matrix(output_size, input_size)
		biases: new_matrix(output_size, 1)
		activation_type: act
		custom_act: CustomActivation{
			forward: dummy_act
			derivative: dummy_act
		}
		last_input: new_matrix(input_size, 1)
		last_output: new_matrix(output_size, 1)
		delta: new_matrix(output_size, 1)
		grad_w: new_matrix(output_size, input_size)
		m_w: new_matrix(output_size, input_size)
		v_w: new_matrix(output_size, input_size)
		m_b: new_matrix(output_size, 1)
		v_b: new_matrix(output_size, 1)
		beta1_t: to_real(1.0)
		beta2_t: to_real(1.0)
		dropout_rate: dropout_rate
		is_rnn: is_rnn
		hidden_weights: if is_rnn {
			new_random_matrix(output_size, output_size)
		} else {
			new_matrix(1, 1)
		}
		prev_hidden: if is_rnn {
			new_matrix(output_size, 1)
		} else {
			new_matrix(1, 1)
		}
		is_custom: false
	}
}

pub fn (mut s Sequential) add_custom(input_size int, output_size int, custom_act CustomActivation, dropout_rate f64, is_rnn bool) {
	s.net.layers << Layer{
		weights: new_random_matrix(output_size, input_size)
		biases: new_matrix(output_size, 1)
		activation_type: .custom
		custom_act: custom_act
		last_input: new_matrix(input_size, 1)
		last_output: new_matrix(output_size, 1)
		delta: new_matrix(output_size, 1)
		grad_w: new_matrix(output_size, input_size)
		m_w: new_matrix(output_size, input_size)
		v_w: new_matrix(output_size, input_size)
		m_b: new_matrix(output_size, 1)
		v_b: new_matrix(output_size, 1)
		beta1_t: to_real(1.0)
		beta2_t: to_real(1.0)
		dropout_rate: dropout_rate
		is_rnn: is_rnn
		hidden_weights: if is_rnn {
			new_random_matrix(output_size, output_size)
		} else {
			new_matrix(1, 1)
		}
		prev_hidden: if is_rnn {
			new_matrix(output_size, 1)
		} else {
			new_matrix(1, 1)
		}
		is_custom: false
	}
}

pub fn (mut s Sequential) add_custom_layer(input_size int, output_size int, forward CustomLayerForwardFn, backward CustomLayerBackwardFn) {
	s.net.layers << Layer{
		weights: new_random_matrix(output_size, input_size)
		biases: new_matrix(output_size, 1)
		activation_type: .linear
		custom_act: CustomActivation{
			forward: dummy_act
			derivative: dummy_act
		}
		last_input: new_matrix(input_size, 1)
		last_output: new_matrix(output_size, 1)
		delta: new_matrix(output_size, 1)
		grad_w: new_matrix(output_size, input_size)
		m_w: new_matrix(output_size, input_size)
		v_w: new_matrix(output_size, input_size)
		m_b: new_matrix(output_size, 1)
		v_b: new_matrix(output_size, 1)
		beta1_t: to_real(1.0)
		beta2_t: to_real(1.0)
		dropout_rate: 0.0
		is_rnn: false
		hidden_weights: new_matrix(1, 1)
		prev_hidden: new_matrix(1, 1)
		is_custom: true
		custom_forward: forward
		custom_backward: backward
	}
}

pub fn (mut s Sequential) set_normalize(val bool) {
	s.net.normalize = val
}

pub fn (mut s Sequential) free() {
	s.net.free()
}

pub fn (mut s Sequential) apply_quantization() {
	if s.quantization == .none {
		return
	}
	limit := if s.quantization == .int8 { to_real(127.0) } else { to_real(32767.0) }
	for mut layer in s.net.layers {
		mut max_w := to_real(0.0)
		for val in layer.weights.data {
			abs_val := if val < to_real(0.0) { -val } else { val }
			if abs_val > max_w {
				max_w = abs_val
			}
		}
		if max_w > to_real(0.0) {
			scale := max_w / limit
			for i in 0 .. layer.weights.data.len {
				q_val := math.round(f64(layer.weights.data[i] / scale))
				mut clipped := if q_val > f64(limit) { f64(limit) } else { q_val }
				clipped = if clipped < f64(-limit) { f64(-limit) } else { clipped }
				layer.weights.data[i] = to_real(clipped) * scale
			}
		}
	}
}

pub fn (s &Sequential) save(path string) ! {
	mut b := new_byte_buffer(1024)
	b.write_u8(u8(`V`))
	b.write_u8(u8(`N`))
	b.write_u8(u8(`M`))
	b.write_u8(u8(`B`))
	b.write_bool(s.net.normalize)
	b.write_u8(u8(s.net.optimizer))
	b.write_u8(u8(s.quantization))
	b.write_int(s.net.means.len)
	unsafe {
		if s.net.means.len > 0 {
			b.write_bytes(&s.net.means[0], s.net.means.len * size_of_real)
		}
	}
	b.write_int(s.net.stds.len)
	unsafe {
		if s.net.stds.len > 0 {
			b.write_bytes(&s.net.stds[0], s.net.stds.len * size_of_real)
		}
	}
	b.write_int(s.net.layers.len)
	for layer in s.net.layers {
		b.write_u8(u8(layer.activation_type))
		b.write_bool(layer.is_rnn)
		b.write_f64(layer.dropout_rate)
		b.write_matrix(layer.weights)
		b.write_matrix(layer.biases)
		b.write_matrix(layer.hidden_weights)
		b.write_matrix(layer.prev_hidden)
		b.write_bool(layer.is_custom)
	}
	mut f := os.create(path)!
	defer { f.close() }
	f.write(b.data)!
}

pub fn load_sequential(path string) !Sequential {
	data := os.read_bytes(path)!
	if data.len < 4 {
		return error("Invalid binary file: Too short.")
	}
	mut r := ByteReader{ data: data, pos: 0 }
	m0 := r.read_u8()
	m1 := r.read_u8()
	m2 := r.read_u8()
	m3 := r.read_u8()
	if m0 != u8(`V`) || m1 != u8(`N`) || m2 != u8(`M`) || m3 != u8(`B`) {
		return error("Invalid binary file: Magic bytes mismatch.")
	}
	normalize := r.read_bool()
	mut optimizer := OptimizerType.sgd
	mut quantization := QuantizationType.none
	unsafe {
		optimizer = OptimizerType(r.read_u8())
		quantization = QuantizationType(r.read_u8())
	}
	means_len := r.read_int()
	mut means := []Real{len: means_len}
	unsafe {
		if means_len > 0 {
			ptr := &u8(&means[0])
			len_bytes := means_len * size_of_real
			for i in 0 .. len_bytes {
				ptr[i] = r.data[r.pos + i]
			}
			r.pos += len_bytes
		}
	}
	stds_len := r.read_int()
	mut stds := []Real{len: stds_len}
	unsafe {
		if stds_len > 0 {
			ptr := &u8(&stds[0])
			len_bytes := stds_len * size_of_real
			for i in 0 .. len_bytes {
				ptr[i] = r.data[r.pos + i]
			}
			r.pos += len_bytes
		}
	}
	layers_len := r.read_int()
	mut layers := []Layer{}
	for _ in 0 .. layers_len {
		mut act_type := ActivationType.sigmoid
		unsafe {
			act_type = ActivationType(r.read_u8())
		}
		is_rnn := r.read_bool()
		dropout_rate := r.read_f64()
		weights := r.read_matrix()
		biases := r.read_matrix()
		hidden_weights := r.read_matrix()
		prev_hidden := r.read_matrix()
		is_custom := r.read_bool()
		input_size := weights.cols
		output_size := weights.rows
		layers << Layer{
			weights: weights
			biases: biases
			activation_type: act_type
			custom_act: CustomActivation{
				forward: dummy_act
				derivative: dummy_act
			}
			last_input: new_matrix(input_size, 1)
			last_output: new_matrix(output_size, 1)
			delta: new_matrix(output_size, 1)
			grad_w: new_matrix(output_size, input_size)
			m_w: new_matrix(output_size, input_size)
			v_w: new_matrix(output_size, input_size)
			m_b: new_matrix(output_size, 1)
			v_b: new_matrix(output_size, 1)
			beta1_t: to_real(1.0)
			beta2_t: to_real(1.0)
			dropout_rate: dropout_rate
			is_rnn: is_rnn
			hidden_weights: hidden_weights
			prev_hidden: prev_hidden
			is_custom: is_custom
			custom_forward: dummy_layer_forward
			custom_backward: dummy_layer_backward
		}
	}
	return Sequential{
		net: NeuralNetwork{
			layers: layers
			optimizer: optimizer
			normalize: normalize
			means: means
			stds: stds
		}
		quantization: quantization
	}
}

@[inline]
fn vnm_log(msg string) {
	$if vnm_safe ? {
		println('[VNM-SAFE] ${msg}')
	}
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) compute_normalization_params(inputs []Tensor) {
	if inputs.len == 0 {
		return
	}
	feat_size := inputs[0].data.len
	unsafe {
		if nn.means.len > 0 {
			nn.means.free()
		}
		if nn.stds.len > 0 {
			nn.stds.free()
		}
	}
	nn.means = []Real{len: feat_size, init: to_real(0.0)}
	nn.stds = []Real{len: feat_size, init: to_real(0.0)}
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
		inv_m := to_real(1.0) / to_real(inputs.len)
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
		eps := to_real(1e-8)
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
pub fn (mut s Sequential) train(inputs []Tensor, targets []Tensor, epochs int, lr f64) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, to_real(lr), to_real(1.0), 0)!
	}
	s.apply_quantization()
}

@[manualfree]
pub fn (mut s Sequential) train_with_decay(inputs []Tensor, targets []Tensor, epochs int, lr f64, decay_rate f64, decay_steps int) ! {
	unsafe {
		s.net.train_with_decay(inputs, targets, epochs, to_real(lr), to_real(decay_rate),
			decay_steps)!
	}
	s.apply_quantization()
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) predict(input Tensor) !Tensor {
	return nn.predict_internal(input, true)!
}

@[manualfree; unsafe]
fn (mut nn NeuralNetwork) forward_pass(input Tensor, perform_normalization bool) ! {
	$if vnm_safe ? {
		if nn.layers.len == 0 {
			return error('Forward pass failed: NeuralNetwork has no layers.')
		}
		first_layer_input_size := nn.layers[0].weights.cols
		if input.data.len != first_layer_input_size {
			return error('Dimension mismatch: Input size is ${input.data.len}, but network expects ${first_layer_input_size}.')
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
			if layer.is_custom {
				layer.last_output = layer.custom_forward(mut *layer, layer.last_input)
				if layer.is_rnn {
					copy_real(&layer.prev_hidden.data[0], &layer.last_output.data[0], layer.last_output.data.len)
				}
				continue
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
					mut i := 0
					for i < len - 3 {
						val0 := ptr_res[i] + ptr_bias[i]
						val1 := ptr_res[i+1] + ptr_bias[i+1]
						val2 := ptr_res[i+2] + ptr_bias[i+2]
						val3 := ptr_res[i+3] + ptr_bias[i+3]
						
						ptr_res[i] = approx_sigmoid(val0)
						ptr_res[i+1] = approx_sigmoid(val1)
						ptr_res[i+2] = approx_sigmoid(val2)
						ptr_res[i+3] = approx_sigmoid(val3)
						i += 4
					}
					for i < len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = approx_sigmoid(val)
						i++
					}
				}
				.relu {
					mut i := 0
					for i < len - 3 {
						val0 := ptr_res[i] + ptr_bias[i]
						val1 := ptr_res[i+1] + ptr_bias[i+1]
						val2 := ptr_res[i+2] + ptr_bias[i+2]
						val3 := ptr_res[i+3] + ptr_bias[i+3]
						
						ptr_res[i] = if val0 > to_real(0.0) { val0 } else { to_real(0.0) }
						ptr_res[i+1] = if val1 > to_real(0.0) { val1 } else { to_real(0.0) }
						ptr_res[i+2] = if val2 > to_real(0.0) { val2 } else { to_real(0.0) }
						ptr_res[i+3] = if val3 > to_real(0.0) { val3 } else { to_real(0.0) }
						i += 4
					}
					for i < len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = if val > to_real(0.0) { val } else { to_real(0.0) }
						i++
					}
				}
				.tanh {
					mut i := 0
					for i < len - 3 {
						val0 := ptr_res[i] + ptr_bias[i]
						val1 := ptr_res[i+1] + ptr_bias[i+1]
						val2 := ptr_res[i+2] + ptr_bias[i+2]
						val3 := ptr_res[i+3] + ptr_bias[i+3]
						
						ptr_res[i] = approx_tanh(val0)
						ptr_res[i+1] = approx_tanh(val1)
						ptr_res[i+2] = approx_tanh(val2)
						ptr_res[i+3] = approx_tanh(val3)
						i += 4
					}
					for i < len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = approx_tanh(val)
						i++
					}
				}
				.linear {
					mut i := 0
					for i < len - 3 {
						ptr_res[i] = ptr_res[i] + ptr_bias[i]
						ptr_res[i+1] = ptr_res[i+1] + ptr_bias[i+1]
						ptr_res[i+2] = ptr_res[i+2] + ptr_bias[i+2]
						ptr_res[i+3] = ptr_res[i+3] + ptr_bias[i+3]
						i += 4
					}
					for i < len {
						ptr_res[i] = ptr_res[i] + ptr_bias[i]
						i++
					}
				}
				.custom {
					mut i := 0
					for i < len - 3 {
						val0 := ptr_res[i] + ptr_bias[i]
						val1 := ptr_res[i+1] + ptr_bias[i+1]
						val2 := ptr_res[i+2] + ptr_bias[i+2]
						val3 := ptr_res[i+3] + ptr_bias[i+3]
						
						ptr_res[i] = layer.custom_act.forward(val0)
						ptr_res[i+1] = layer.custom_act.forward(val1)
						ptr_res[i+2] = layer.custom_act.forward(val2)
						ptr_res[i+3] = layer.custom_act.forward(val3)
						i += 4
					}
					for i < len {
						val := ptr_res[i] + ptr_bias[i]
						ptr_res[i] = layer.custom_act.forward(val)
						i++
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
			return error('Training failed: NeuralNetwork has no layers.')
		}
		last_layer_idx := nn.layers.len - 1
		expected_output_size := nn.layers[last_layer_idx].weights.rows
		if target.data.len != expected_output_size {
			return error('Dimension mismatch: Target size is ${target.data.len}, but network expects ${expected_output_size}.')
		}
	}
	unsafe {
		nn.forward_pass(input, !is_normalized)!
		mut output_layer := &nn.layers[nn.layers.len - 1]
		mut step_loss := to_real(0.0)
		target_matrix := Matrix{
			rows: target.data.len
			cols: 1
			data: target.data
		}
		mut loss_grad := new_matrix(output_layer.last_output.rows, output_layer.last_output.cols)
		nn.loss_fn.gradient(output_layer.last_output, target_matrix, mut loss_grad)
		step_loss = nn.loss_fn.compute(output_layer.last_output, target_matrix)
		
		mut ptr_delta := &output_layer.delta.data[0]
		mut ptr_out := &output_layer.last_output.data[0]
		mut ptr_grad := &loss_grad.data[0]
		len := output_layer.delta.data.len
		match output_layer.activation_type {
			.sigmoid {
				for i in 0 .. len {
					ptr_delta[i] = ptr_grad[i] * (ptr_out[i] * (to_real(1.0) - ptr_out[i]))
				}
			}
			.relu {
				for i in 0 .. len {
					ptr_delta[i] = if ptr_out[i] > to_real(0.0) { ptr_grad[i] } else { to_real(0.0) }
				}
			}
			.tanh {
				for i in 0 .. len {
					ptr_delta[i] = ptr_grad[i] * (to_real(1.0) - (ptr_out[i] * ptr_out[i]))
				}
			}
			.linear {
				for i in 0 .. len {
					ptr_delta[i] = ptr_grad[i]
				}
			}
			.custom {
				for i in 0 .. len {
					ptr_delta[i] = ptr_grad[i] * output_layer.custom_act.derivative(ptr_out[i])
				}
			}
		}
		for l := nn.layers.len - 1; l >= 0; l-- {
			mut current_layer := &nn.layers[l]
			if current_layer.is_custom {
				if l > 0 {
					mut prev_layer := &nn.layers[l - 1]
					prev_layer.delta = current_layer.custom_backward(mut *current_layer, current_layer.delta)
				} else {
					_ = current_layer.custom_backward(mut *current_layer, current_layer.delta)
				}
			} else {
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
								ptr_next[i] = ptr_next[i] * (ptr_prev_out[i] * (to_real(1.0) - ptr_prev_out[i]))
							}
						}
						.relu {
							for i in 0 .. len_prev {
								ptr_next[i] = if ptr_prev_out[i] > to_real(0.0) { ptr_next[i] } else { to_real(0.0) }
							}
						}
						.tanh {
							for i in 0 .. len_prev {
								ptr_next[i] = ptr_next[i] * (to_real(1.0) - (ptr_prev_out[i] * ptr_prev_out[i]))
							}
						}
						.linear {}
						.custom {
							for i in 0 .. len_prev {
								ptr_next[i] = ptr_next[i] * prev_layer.custom_act.derivative(ptr_prev_out[i])
							}
						}
					}
				}
			}
			if current_layer.weights.data.len > 0 {
				if _likely_(nn.optimizer == .adam) {
					beta1 := to_real(0.9)
					beta2 := to_real(0.999)
					mut eps_sq := to_real(0.0)
					$if vnm_f64 ? {
						eps_sq = to_real(1e-12)
					} $else {
						eps_sq = to_real(1e-8)
					}
					current_layer.beta1_t *= beta1
					current_layer.beta2_t *= beta2
					bias_correction1 := to_real(1.0) - current_layer.beta1_t
					bias_correction2 := to_real(1.0) - current_layer.beta2_t
					mut ptr_w := &current_layer.weights.data[0]
					mut ptr_g := &current_layer.grad_w.data[0]
					mut ptr_mw := &current_layer.m_w.data[0]
					mut ptr_vw := &current_layer.v_w.data[0]
					one_minus_beta1 := to_real(1.0) - beta1
					one_minus_beta2 := to_real(1.0) - beta2
					step_size := lr / bias_correction1
					inv_bias_corr2 := to_real(1.0) / bias_correction2
					len_w := current_layer.weights.data.len
					for i in 0 .. len_w {
						mw_val := beta1 * ptr_mw[i] + one_minus_beta1 * ptr_g[i]
						vw_val := beta2 * ptr_vw[i] + one_minus_beta2 * ptr_g[i] * ptr_g[i]
						ptr_mw[i] = mw_val
						ptr_vw[i] = vw_val
						v_hat := vw_val * inv_bias_corr2
						ptr_w[i] -= step_size * mw_val * approx_inv_sqrt(v_hat + eps_sq)
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
						ptr_b[i] -= step_size * mb_val * approx_inv_sqrt(v_hat_b + eps_sq)
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
		}
		loss_grad.free()
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
		if _unlikely_(decay_rate < to_real(1.0) && decay_steps > 0 && epoch > 0 && epoch % decay_steps == 0) {
			current_lr *= decay_rate
		}
		mut total_error := to_real(0.0)
		for i in 0 .. training_inputs.len {
			step_loss := nn.train_step_internal(training_inputs[i], targets[i], current_lr, true)!
			total_error += step_loss
		}
		if epochs >= 10 && epoch % (epochs / 10) == 0 {
			$if !vnm_silent ? {
				mean_error := total_error / to_real(inputs.len)
				println('  Epoch ${epoch:5d} / ${epochs} | Active LR: ${current_lr:.6f} | Mean Squared Loss: ${mean_error:.8f}')
			}
		}
	}
	if nn.normalize {
		for mut t in temp_normalized_tensors {
			t.free()
		}
		unsafe {
			temp_normalized_tensors.free()
		}
	}
}

@[manualfree; unsafe]
pub fn (mut nn NeuralNetwork) train_step(input Tensor, target Tensor, lr f64) !Tensor {
	unsafe {
		_ = nn.train_step_internal(input, target, to_real(lr), false)!
		last_layer := &nn.layers[nn.layers.len - 1]
		return Tensor{
			shape: [last_layer.last_output.data.len]
			data: last_layer.last_output.data.clone()
		}
	}
}

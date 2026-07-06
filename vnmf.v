module vnm

import os

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

fn C.open(pathname &char, flags int, mode int) int
fn C.close(fd int) int
fn C.mmap(addr voidptr, length usize, prot int, flags int, fd int, offset i64) voidptr
fn C.munmap(addr voidptr, length usize) int
fn C.madvise(addr voidptr, length usize, advice int) int

const prot_read = C.PROT_READ
const map_private = C.MAP_PRIVATE
const madv_willneed = C.MADV_WILLNEED
const madv_dontneed = C.MADV_DONTNEED

pub struct FocusEngine {
pub mut:
	fd        int
	file_size u64
	mapped    voidptr
}

pub struct MappedModel {
pub mut:
	model  Sequential
	engine FocusEngine
}

pub fn (mut mm MappedModel) free() {
	mm.model.free()
	mm.engine.unmap()
}

struct MappedReader {
	mapped voidptr
	size   u64
mut:
	pos    u64
}

fn (mut r MappedReader) read_u8() u8 {
	unsafe {
		val := *(&u8(voidptr(u64(r.mapped) + r.pos)))
		r.pos++
		return val
	}
}

fn (mut r MappedReader) read_bool() bool {
	return r.read_u8() == 1
}

fn (mut r MappedReader) read_int() int {
	unsafe {
		val := *(&int(voidptr(u64(r.mapped) + r.pos)))
		r.pos += 4
		return val
	}
}

fn (mut r MappedReader) read_f64() f64 {
	unsafe {
		val := *(&f64(voidptr(u64(r.mapped) + r.pos)))
		r.pos += 8
		return val
	}
}

fn (mut r MappedReader) read_matrix_view() Matrix {
	rows := r.read_int()
	cols := r.read_int()
	unsafe {
		ptr := voidptr(u64(r.mapped) + r.pos)
		r.pos += u64(rows * cols * size_of_real)
		mut data := []Real{}
		data.data = ptr
		data.len = rows * cols
		data.cap = rows * cols
		return Matrix{
			rows: rows
			cols: cols
			data: data
		}
	}
}

pub fn load_sequential_mapped(path string) !MappedModel {
	mut engine := map_file(path)!
	engine.prefetch(0, engine.file_size)
	model := engine.parse_to_sequential()!
	return MappedModel{
		model: model
		engine: engine
	}
}

fn (f &FocusEngine) parse_to_sequential() !Sequential {
	if f.file_size < 4 {
		return error("File too short")
	}
	mut r := MappedReader{
		mapped: f.mapped
		size: f.file_size
		pos: 0
	}
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
	mut means := []Real{}
	if means_len > 0 {
		unsafe {
			means.data = voidptr(u64(r.mapped) + r.pos)
			means.len = means_len
			means.cap = means_len
		}
		r.pos += u64(means_len * size_of_real)
	}

	stds_len := r.read_int()
	mut stds := []Real{}
	if stds_len > 0 {
		unsafe {
			stds.data = voidptr(u64(r.mapped) + r.pos)
			stds.len = stds_len
			stds.cap = stds_len
		}
		r.pos += u64(stds_len * size_of_real)
	}

	layers_len := r.read_int()
	mut layers := []Layer{}
	for _ in 0 .. layers_len {
		mut act_type := ActivationType.sigmoid
		unsafe { act_type = ActivationType(r.read_u8()) }
		is_rnn := r.read_bool()
		dropout_rate := r.read_f64()

		weights := r.read_matrix_view()
		biases := r.read_matrix_view()
		hidden_weights := r.read_matrix_view()
		prev_hidden := r.read_matrix_view()

		is_custom := r.read_bool()
		input_size := weights.cols
		output_size := weights.rows

		layers << Layer{
			weights: weights
			biases: biases
			activation_type: act_type
			custom_act: CustomActivation{ forward: dummy_act, derivative: dummy_act }
			last_input: new_matrix(input_size, 1)
			last_output: new_matrix(output_size, 1)
			delta: Matrix{}
			grad_w: Matrix{}
			m_w: Matrix{}
			v_w: Matrix{}
			m_b: Matrix{}
			v_b: Matrix{}
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

pub fn map_file(path string) !FocusEngine {
	$if windows {
		return error("Memory mapping is only supported on POSIX systems")
	} $else {
		if !os.exists(path) {
			return error("File not found")
		}
		size := os.file_size(path)
		fd := C.open(&char(path.str), 0, 0)
		if fd < 0 {
			return error("Failed to open file")
		}
		mapped := C.mmap(unsafe { nil }, usize(size), prot_read, map_private, fd, 0)
		if mapped == voidptr(-1) {
			C.close(fd)
			return error("Failed to map file")
		}
		return FocusEngine{
			fd: fd
			file_size: size
			mapped: mapped
		}
	}
}

pub fn (mut f FocusEngine) unmap() {
	$if !windows {
		if f.mapped != voidptr(0) && f.mapped != voidptr(-1) {
			C.munmap(f.mapped, usize(f.file_size))
			f.mapped = voidptr(0)
		}
		if f.fd >= 0 {
			C.close(f.fd)
			f.fd = -1
		}
	}
}

pub fn (f &FocusEngine) prefetch(offset u64, length_bytes u64) {
	$if !windows {
		unsafe {
			ptr := voidptr(u64(f.mapped) + offset)
			C.madvise(ptr, usize(length_bytes), madv_willneed)
		}
	}
}

pub fn (f &FocusEngine) evict(offset u64, length_bytes u64) {
	$if !windows {
		unsafe {
			ptr := voidptr(u64(f.mapped) + offset)
			C.madvise(ptr, usize(length_bytes), madv_dontneed)
		}
	}
}

pub fn (f &FocusEngine) get_matrix(offset u64, rows int, cols int) Matrix {
	unsafe {
		ptr := &Real(voidptr(u64(f.mapped) + offset))
		mut data := []Real{len: rows * cols}
		C.memcpy(&data[0], ptr, usize(rows * cols * size_of_real))
		return Matrix{
			rows: rows
			cols: cols
			data: data
		}
	}
}

pub fn (f &FocusEngine) get_matrix_view(offset u64, rows int, cols int) Matrix {
	unsafe {
		ptr := &Real(voidptr(u64(f.mapped) + offset))
		mut res := Matrix{
			rows: rows
			cols: cols
		}
		mut data := []Real{}
		data.data = voidptr(ptr)
		data.len = rows * cols
		data.cap = rows * cols
		res.data = data
		return res
	}
}

pub fn (f &FocusEngine) parse_topology() ![]u64 {
	mut offsets := []u64{}
	unsafe {
		mut pos := u64(0)
		if f.file_size < 11 {
			return error("File too short")
		}
		pos += 4
		pos += 1
		pos += 1
		pos += 1
		mut ptr := &u8(voidptr(u64(f.mapped) + pos))
		means_len := *(&int(voidptr(ptr)))
		pos += 4
		pos += u64(means_len * size_of_real)
		ptr = &u8(voidptr(u64(f.mapped) + pos))
		stds_len := *(&int(voidptr(ptr)))
		pos += 4
		pos += u64(stds_len * size_of_real)
		ptr = &u8(voidptr(u64(f.mapped) + pos))
		layers_len := *(&int(voidptr(ptr)))
		pos += 4
		for _ in 0 .. layers_len {
			pos += 1
			pos += 1
			pos += 8
			offsets << pos
			ptr = &u8(voidptr(u64(f.mapped) + pos))
			w_rows := *(&int(voidptr(ptr)))
			w_cols := *(&int(voidptr(ptr + 4)))
			pos += 8 + u64(w_rows * w_cols * size_of_real)
			ptr = &u8(voidptr(u64(f.mapped) + pos))
			b_rows := *(&int(voidptr(ptr)))
			b_cols := *(&int(voidptr(ptr + 4)))
			pos += 8 + u64(b_rows * b_cols * size_of_real)
			ptr = &u8(voidptr(u64(f.mapped) + pos))
			hw_rows := *(&int(voidptr(ptr)))
			hw_cols := *(&int(voidptr(ptr + 4)))
			pos += 8 + u64(hw_rows * hw_cols * size_of_real)
			ptr = &u8(voidptr(u64(f.mapped) + pos))
			ph_rows := *(&int(voidptr(ptr)))
			ph_cols := *(&int(voidptr(ptr + 4)))
			pos += 8 + u64(ph_rows * ph_cols * size_of_real)
			pos += 1
		}
	}
	return offsets
}

pub fn (f &FocusEngine) prefetch_async(offset u64, length_bytes u64) {
	spawn f.prefetch_worker(offset, length_bytes)
}

fn (f &FocusEngine) prefetch_worker(offset u64, length_bytes u64) {
	f.prefetch(offset, length_bytes)
}

pub fn (f &FocusEngine) get_layer_sizes(offsets []u64) []u64 {
	mut sizes := []u64{cap: offsets.len}
	for i in 0 .. offsets.len {
		if i < offsets.len - 1 {
			sizes << offsets[i + 1] - offsets[i]
		} else {
			sizes << f.file_size - offsets[i]
		}
	}
	return sizes
}

pub fn (f &FocusEngine) prefetch_layer(offsets []u64, sizes []u64, layer_idx int) {
	if layer_idx >= 0 && layer_idx < offsets.len {
		f.prefetch(offsets[layer_idx], sizes[layer_idx])
	}
}

pub fn (f &FocusEngine) prefetch_layer_async(offsets []u64, sizes []u64, layer_idx int) {
	if layer_idx >= 0 && layer_idx < offsets.len {
		f.prefetch_async(offsets[layer_idx], sizes[layer_idx])
	}
}

pub fn (f &FocusEngine) evict_layer(offsets []u64, sizes []u64, layer_idx int) {
	if layer_idx >= 0 && layer_idx < offsets.len {
		f.evict(offsets[layer_idx], sizes[layer_idx])
	}
}

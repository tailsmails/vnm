import vnm
import math
import os

fn simulate_orbit(a f64, e f64, t f64) (f64, f64, f64, f64) {
	mu := 398600.44
	n := math.sqrt(mu / (a * a * a))
	mut m := n * t
	
	m = math.mod(m, 2.0 * math.pi)
	if m < 0.0 {
		m += 2.0 * math.pi
	}

	mut ecc_anomaly := m
	for _ in 0 .. 6 {
		ecc_anomaly = ecc_anomaly - (ecc_anomaly - e * math.sin(ecc_anomaly) - m) / (1.0 - e * math.cos(ecc_anomaly))
	}

	x_orb := a * (math.cos(ecc_anomaly) - e)
	y_orb := a * math.sqrt(1.0 - e * e) * math.sin(ecc_anomaly)

	r := a * (1.0 - e * math.cos(ecc_anomaly))
	vx_orb := -math.sqrt(mu * a) / r * math.sin(ecc_anomaly)
	vy_orb := math.sqrt(mu * a * (1.0 - e * e)) / r * math.cos(ecc_anomaly)

	return x_orb, y_orb, vx_orb, vy_orb
}

fn main() {
	exe_dir := os.dir(os.executable())
	model_path := os.join_path(exe_dir, "satellite_orbit_model.vnm")
	mut model := vnm.new_sequential(.adam)

	if os.exists(model_path) {
		model = vnm.load_sequential(model_path) or {
			println("Failed to load model: ${err}")
			return
		}
	} else {
		model.add(3, 128, .relu)
		model.add(128, 64, .relu)
		model.add(64, 4, .linear)

		model.set_normalize(true)

		mut inputs := []vnm.Tensor{}
		mut targets := []vnm.Tensor{}

		train_a := [7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0]
		train_e := [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

		for a in train_a {
			for e in train_e {
				for t_step in 0 .. 21 {
					t := f64(t_step) * 500.0
					x, y, vx, vy := simulate_orbit(a, e, t)
					
					inputs << vnm.vector([a, e, t])
					
					norm_x := x / 15000.0
					norm_y := y / 15000.0
					norm_vx := vx / 15.0
					norm_vy := vy / 15.0
					targets << vnm.vector([norm_x, norm_y, norm_vx, norm_vy])
				}
			}
		}

		model.train_with_decay(inputs, targets, 1500, 0.0008, 1.0, 0) or { return }
		model.train_with_decay(inputs, targets, 1000, 0.0005, 0.94, 200) or { return }
		model.save(model_path) or { return }
	}

	blind_cases := [
		[7800.0, 0.15, 2300.0],
		[8800.0, 0.25, 6200.0],
		[9300.0, 0.05, 8800.0],
		[7200.0, 0.35, 1200.0]
	]

	for test in blind_cases {
		a := test[0]
		e := test[1]
		t := test[2]

		real_x, real_y, real_vx, real_vy := simulate_orbit(a, e, t)

		pred_tensor := model.predict(vnm.vector([a, e, t])) or {
			vnm.vector([0.0, 0.0, 0.0, 0.0])
		}

		pred_x := pred_tensor.data[0] * 15000.0
		pred_y := pred_tensor.data[1] * 15000.0
		pred_vx := pred_tensor.data[2] * 15.0
		pred_vy := pred_tensor.data[3] * 15.0

		err_x := math.abs(real_x - pred_x)
		err_y := math.abs(real_y - pred_y)
		err_vx := math.abs(real_vx - pred_vx)
		err_vy := math.abs(real_vy - pred_vy)

		println("Inputs -> Semi-major Axis (a): ${a:5.0f}km | Eccentricity (e): ${e:3.2f} | Time (t): ${t:5.0f}s")
		println("  Real:      Pos = (${real_x:7.1f}, ${real_y:7.1f}) km | Vel = (${real_vx:5.2f}, ${real_vy:5.2f}) km/s")
		println("  Predicted: Pos = (${pred_x:7.1f}, ${pred_y:7.1f}) km | Vel = (${pred_vx:5.2f}, ${pred_vy:5.2f}) km/s")
		println("  Abs Error: ΔPos = (${err_x:5.1f}, ${err_y:5.1f}) km | ΔVel = (${err_vx:4.2f}, ${err_vy:4.2f}) km/s")
		println("")
	}
}
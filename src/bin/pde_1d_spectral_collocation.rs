use plotpy::{Curve, Plot};
use russell_lab::algo::{InterpLagrange, InterpParams};
use russell_lab::math::NAPIER;
use russell_lab::{solve_lin_sys, vec_max_abs_diff, Matrix, StrError, Vector};

const PATH_KEY: &str = "/tmp/examples_russell/pde_1d_spectral_collocation";

// LORENE example (page 25 of Reference)
//
// PDE:
//
// d²u     du          x
// ——— - 4 —— + 4 u = e  + C
// dx²     dx
//
//     -4 e
// C = ——————
//     1 + e²
//
// x ∈ [-1, 1]
//
// Boundary conditions:
//
// u(-1) = 0  and  u(1) = 0
//
// Solution:
//
//         x   sinh(1)  2x   C
// u(x) = e  - ——————— e   + —
//             sinh(2)       4
//
// Reference:
// * Gourgoulhon E (2005), An introduction to polynomial interpolation,
//   School on spectral methods: Application to General Relativity and Field Theory
//   Meudon, 14-18 November 2005

fn main() -> Result<(), StrError> {
    // analytical solution
    const CC: f64 = -4.0 * NAPIER / (1.0 + NAPIER * NAPIER);
    let sh1 = f64::sinh(1.0);
    let sh2 = f64::sinh(2.0);
    let solution = |x| f64::exp(x) - f64::exp(2.0 * x) * sh1 / sh2 + CC / 4.0;

    // interpolant
    let nn = 4;
    let params = InterpParams::new();
    let mut interp = InterpLagrange::new(nn, Some(params))?;

    // D1 and D2 matrices
    interp.calc_dd1_matrix();
    interp.calc_dd2_matrix();
    let dd1 = interp.get_dd1()?;
    let dd2 = interp.get_dd2()?;

    // source term (right-hand side)
    let xx = interp.get_points();
    let npoint = xx.dim();
    let mut b = Vector::new(npoint);
    for i in 0..npoint {
        b[i] = f64::exp(xx[i]) + CC;
    }
    b[0] = 0.0;
    b[nn] = 0.0;

    // discrete differential operator
    let mut aa = Matrix::new(npoint, npoint);
    for i in 1..nn {
        for j in 1..nn {
            let o = if i == j { 1.0 } else { 0.0 };
            aa.set(i, j, dd2.get(i, j) - 4.0 * dd1.get(i, j) + 4.0 * o);
        }
    }
    aa.set(0, 0, 1.0);
    aa.set(nn, nn, 1.0);

    // linear system
    solve_lin_sys(&mut b, &mut aa)?;
    let uu = &b;

    // error at nodes
    let uu_ana = xx.get_mapped(solution);
    let max_diff = vec_max_abs_diff(&uu, &uu_ana)?;
    println!("max diff = {}", max_diff.1);

    // plot
    let mut curve_num1 = Curve::new();
    let mut curve_num2 = Curve::new();
    let mut curve_ana = Curve::new();
    let xx_plt = Vector::linspace(-1.0, 1.0, 201)?;
    let yy_num = xx_plt.get_mapped(|x| interp.eval(x, uu).unwrap());
    let yy_ana = xx_plt.get_mapped(solution);
    curve_ana
        .set_label("analytical")
        .draw(xx_plt.as_data(), yy_ana.as_data());
    curve_num1
        .set_label("numerical")
        .draw(xx_plt.as_data(), yy_num.as_data());
    curve_num2
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true)
        .draw(xx.as_data(), uu.as_data());
    let mut plot = Plot::new();
    plot.add(&curve_ana)
        .add(&curve_num1)
        .add(&curve_num2)
        .legend()
        .grid_and_labels("$x$", "$u(x)$")
        .set_title(format!("N = {}", nn).as_str())
        .save(format!("{}.svg", PATH_KEY).as_str())?;

    Ok(())
}

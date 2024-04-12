use plotpy::{Curve, Plot};
use russell_lab::algo::InterpLagrange;
use russell_lab::math::PI;
use russell_lab::{mat_vec_mul, StrError, Vector};
use russell_ode::{no_jacobian, HasJacobian, Method, OdeSolver, Params, System};

struct Args {
    interp: InterpLagrange,
}

fn main() -> Result<(), StrError> {
    // polynomial degree and number of points
    let nn = 8;
    let npoint = nn + 1;

    // ODE system
    let ndim = npoint;
    let system = System::new(
        ndim,
        |dudt: &mut Vector, _: f64, u: &Vector, args: &mut Args| {
            mat_vec_mul(dudt, 1.0, args.interp.get_dd2(), u)?;
            dudt[0] = 0.0; // homogeneous boundary conditions
            dudt[nn] = 0.0; // homogeneous boundary conditions
            Ok(())
        }, //
        no_jacobian,
        HasJacobian::No,
        None,
        None,
    );

    // ODE solver
    let params = Params::new(Method::DoPri8);
    let mut ode = OdeSolver::new(params, &system)?;

    // arguments for the system
    let mut args = Args {
        interp: InterpLagrange::new(nn, None)?,
    };

    // initial conditions
    let (t0, t1) = (0.0, 0.1);
    let xx = args.interp.get_points().clone();
    let mut uu = xx.get_mapped(|x| f64::sin(PI * (x + 1.0)));

    // solve the problem
    ode.solve(&mut uu, t0, t1, None, None, &mut args)?;

    // print stats
    println!("{}", ode.stats());

    // plot the results @ t1
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let xx_ana = Vector::linspace(-1.0, 1.0, 201)?;
    let uu_ana = xx_ana.get_mapped(|x| -f64::exp(-PI * PI * t1) * f64::sin(PI * x));
    curve1.draw(xx_ana.as_data(), uu_ana.as_data());
    curve2
        .set_line_style("None")
        .set_marker_style("o")
        .set_marker_void(true)
        .draw(xx.as_data(), uu.as_data());
    let mut plot = Plot::new();
    plot.add(&curve1)
        .add(&curve2)
        .grid_and_labels("$x$", "$u(x)$")
        .save("/tmp/examples_russell/pde_1d_heat_spectral_collocation.svg")?;
    Ok(())
}

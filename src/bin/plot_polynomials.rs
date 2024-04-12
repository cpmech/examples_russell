use plotpy::{Curve, Plot, RayEndpoint};
use russell_lab::algo::*;
use russell_lab::{StrError, Vector};

fn do_plot(grid_type: InterpGrid) -> Result<(), StrError> {
    // parameters
    let mut params = InterpParams::new();
    params.grid_type = grid_type;

    // interpolant
    let nn = 6;
    let interp = InterpLagrange::new(nn, Some(params))?;
    let grid = format!("{:?}", grid_type);
    println!("{}: X =\n{}", grid, Vector::from(interp.get_points()));

    // x coordinate for plotting
    let nstation = 201;
    let stations = Vector::linspace(-1.0, 1.0, nstation)?;

    // y coordinate for plotting
    let mut yy = Vector::new(nstation);
    let mut curve = Curve::new();

    // draw psi
    let npoint = nn + 1;
    for p in 0..npoint {
        for s in 0..nstation {
            yy[s] = interp.psi(p, stations[s])?;
        }
        curve.draw(stations.as_data(), yy.as_data());
    }

    // save figure
    let mut plot = Plot::new();
    let path = format!(
        "/tmp/examples_russell/plot_polynomials_{}.svg",
        grid.to_lowercase()
    );
    let mut line = Curve::new();
    line.set_line_style("--")
        .set_line_color("black")
        .draw_ray(-1.0, 1.0, RayEndpoint::Horizontal);
    plot.add(&line)
        .add(&curve)
        .grid_and_labels("$x$", "$\\psi(x)$")
        .save(path.as_str())?;
    Ok(())
}

fn main() -> Result<(), StrError> {
    do_plot(InterpGrid::Uniform)?;
    do_plot(InterpGrid::ChebyshevGauss)?;
    do_plot(InterpGrid::ChebyshevGaussLobatto)?;
    Ok(())
}

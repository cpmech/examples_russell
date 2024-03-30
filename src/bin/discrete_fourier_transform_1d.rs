use num_complex::Complex64;
use plotpy::{Curve, Plot, RayEndpoint, SuperTitleParams, Text};
use rand::prelude::StdRng;
use rand::SeedableRng;
use russell_lab::{cpx, math::PI, ComplexVector, FFTw};
use russell_stat::{DistributionNormal, ProbabilityDistribution, StrError};

fn main() -> Result<(), StrError> {
    // a uniformly distributed value between 0 and 1:
    let mut rng = StdRng::seed_from_u64(1234);
    let dist = DistributionNormal::new(0.0, 2.0)?;

    // constants
    const F: f64 = 1000.0; // sampling frequency
    const T: f64 = 1.0 / F; // sampling period
    const L: usize = 1500; // length of signal

    // generate a signal with a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz sinusoid of amplitude 1.0
    let mut t = vec![0.0; L]; // time vector
    let mut u_ori = vec![0.0; L]; // original signal
    let mut u_alt = vec![0.0; L]; // altered signal with zero-mean noise a std-dev of 2.0
    let mut z_ori = ComplexVector::new(L); // complex array
    let mut z_alt = ComplexVector::new(L); // complex array
    for i in 0..L {
        t[i] = (i as f64) * T;
        u_ori[i] = 0.7 * f64::sin(2.0 * PI * 50.0 * t[i]) + f64::sin(2.0 * PI * 120.0 * t[i]);
        u_alt[i] = u_ori[i] + 2.0 * dist.sample(&mut rng);
        z_ori[i] = cpx!(u_ori[i], 0.0);
        z_alt[i] = cpx!(u_alt[i], 0.0);
    }

    // perform the DFT on the original signal
    let mut fft = FFTw::new();
    let mut zz_ori = ComplexVector::new(L);
    fft.dft_1d(&mut zz_ori, &z_ori, false)?;

    // perform the DFT on the altered signal
    let mut zz_alt = ComplexVector::new(L);
    fft.dft_1d(&mut zz_alt, &z_alt, false)?;

    // process the results
    const M: usize = L / 2 + 1;
    let mut pp = vec![0.0; M]; // single-sided spectrum of the original signal
    let mut qq = vec![0.0; M]; // single-sided spectrum of the corrupted signal
    let mut ff = vec![0.0; M]; // frequency domain f
    let den = L as f64;
    for i in 0..M {
        pp[i] = 2.0 * zz_ori[i].norm() / den;
        qq[i] = 2.0 * zz_alt[i].norm() / den;
        ff[i] = F * (i as f64) / den;
    }

    // plot
    let mut curve1 = Curve::new();
    let mut curve2 = Curve::new();
    let mut curve3 = Curve::new();
    let mut curve4 = Curve::new();
    let mut curve5 = Curve::new();
    let mut text1 = Text::new();
    curve1
        .set_label("Original signal")
        .set_line_color("blue")
        .set_line_width(2.0);
    curve2.set_label("Altered signal").set_line_color("red");
    curve4
        .set_label("Original signal")
        .set_line_color("blue")
        .set_line_width(2.0);
    curve5.set_label("Altered signal").set_line_color("red");
    curve1.draw(&&t[0..50], &&u_ori[0..50]);
    curve2.draw(&&t[0..50], &&u_alt[0..50]);
    curve3.set_line_style("--").set_line_color("green");
    curve3.draw_ray(0.0, 0.7, RayEndpoint::Horizontal);
    curve3.draw_ray(0.0, 1.0, RayEndpoint::Horizontal);
    curve4.draw(&ff, &pp);
    curve5.draw(&ff, &qq);
    text1.draw(52.0, 0.02, "50 Hz");
    text1.draw(122.0, 0.02, "120 Hz");
    let mut params = SuperTitleParams::new();
    params.set_align_vertical("bottom").set_y(0.9);
    let mut plot = Plot::new();
    plot.set_subplot(3, 1, 1)
        .add(&curve1)
        .add(&curve2)
        .grid_labels_legend("$t\\,[\\mu s]$", "$u$")
        .set_subplot(3, 1, 2)
        .add(&curve3)
        .add(&curve4)
        .add(&text1)
        .grid_labels_legend("$f\\,[Hz]$", "spectrum")
        .set_subplot(3, 1, 3)
        .add(&curve3)
        .add(&curve5)
        .grid_labels_legend("$f\\,[Hz]$", "spectrum")
        .set_figure_size_points(600.0, 800.0)
        .set_super_title(
            "Signal with a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz sinusoid of amplitude 1.0",
            Some(params),
        )
        .save("/tmp/examples_russell/discrete_fourier_transform_1d.svg")?;
    Ok(())
}
